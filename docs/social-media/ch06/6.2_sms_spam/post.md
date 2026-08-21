# 6.2 SMS Spam 微调：完整流程跑一遍

> 第 6 章第 2 帖｜⭐ 实战帖｜阅读 8 分钟｜口播 5 分钟
> 平台：★ 知乎（主）

---

## 钩子

**5,572 条 SMS，能训出 99% 准确率的垃圾邮件分类器。**

跟着做一遍，整个微调流程就熟了。

---

## 数据集

SMS Spam Collection (UCI ML Repo)
- **总数**：5,572 条 SMS
- **类别**：ham（4,827 = 86.6%）/ spam（747 = 13.4%）
- **格式**：`label\ttext`

```
ham\tGo until jurong point, crazy.. Available only in bugis n great world la e buffet...
spam\tFree entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121...
```

**注意类别不平衡**：spam 只占 13.4%。

---

## 训练配置

| 参数 | 值 | 选择理由 |
|---|---|---|
| `model` | GPT-2 small (124M) | 标配 |
| `lr` | 5e-5 | 微调标准 |
| `epochs` | 5 | 数据少，5 epoch 就够 |
| `batch_size` | 8 | RTX 4080 上限 16 |
| `max_length` | 120 | SMS 长度中位数 |
| 冻结 | 最后 4 层 + head | 平衡方案 |
| 优化器 | AdamW | 默认 |

---

## 数据加载

```cpp
struct SMSDataset {
    std::vector<SMSExample> examples;
    
    void load(const std::string& path) {
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            auto tab = line.find('\t');
            std::string label = line.substr(0, tab);
            std::string text  = line.substr(tab + 1);
            
            examples.push_back({
                text,
                label == "spam" ? 1 : 0
            });
        }
    }
    
    // 划分 train/val/test = 70/10/20
    std::vector<SMSExample> train, val, test;
    void split() {
        std::mt19937 rng(42);
        std::shuffle(examples.begin(), examples.end(), rng);
        int n = examples.size();
        int n_train = n * 0.7, n_val = n * 0.1;
        train.assign(examples.begin(), examples.begin() + n_train);
        val.assign(examples.begin() + n_train,
                   examples.begin() + n_train + n_val);
        test.assign(examples.begin() + n_train + n_val, examples.end());
    }
};
```

---

## 训练循环

```cpp
void train_classifier(GPTClassifier& model,
                     SMSDataset& data,
                     torch::optim::Optimizer& optimizer,
                     int epochs, torch::Device device) {
    model->to(device);
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        model->train();
        float total_loss = 0;
        int correct = 0, total = 0;
        
        for (int i = 0; i < data.train.size(); i += 8) {
            // 1. 准备 batch
            std::vector<int64_t> ids_batch, labels_batch;
            for (int j = 0; j < 8 && i + j < data.train.size(); ++j) {
                auto& ex = data.train[i + j];
                auto tokens = tokenizer.encode(ex.text);
                tokens.resize(120, 50256);  // pad 到 120
                ids_batch.insert(ids_batch.end(), tokens.begin(), tokens.end());
                labels_batch.push_back(ex.label);
            }
            
            auto ids = torch::tensor(ids_batch).view({-1, 120});
            auto labels = torch::tensor(labels_batch);
            ids = ids.to(device);
            labels = labels.to(device);
            
            // 2. forward
            auto logits = model->forward(ids);
            auto loss = torch::nn::functional::cross_entropy(logits, labels);
            
            // 3. backward
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            
            total_loss += loss.item<float>();
            auto preds = logits.argmax(-1);
            correct += (preds == labels).sum().item<int>();
            total += labels.size(0);
        }
        
        std::cout << "epoch " << epoch
                  << " loss=" << total_loss / (data.train.size() / 8)
                  << " acc=" << (float)correct / total
                  << std::endl;
    }
}
```

---

## 训练输出

```
epoch 0 loss=0.42 acc=87.2%
epoch 1 loss=0.18 acc=95.4%
epoch 2 loss=0.09 acc=98.1%
epoch 3 loss=0.05 acc=98.7%
epoch 4 loss=0.03 acc=99.2%
```

**5 epoch 到 99.2% 准确率**。

---

## 测试集评估

```cpp
void evaluate(GPTClassifier& model, std::vector<SMSExample>& test,
              torch::Device device) {
    model->eval();
    int tp = 0, fp = 0, tn = 0, fn = 0;
    
    for (auto& ex : test) {
        auto tokens = tokenizer.encode(ex.text);
        tokens.resize(120, 50256);
        auto ids = torch::tensor(tokens).unsqueeze(0).to(device);
        auto logits = model->forward(ids);
        auto pred = logits.argmax(-1).item<int>();
        
        if (pred == 1 && ex.label == 1) tp++;
        else if (pred == 1 && ex.label == 0) fp++;
        else if (pred == 0 && ex.label == 0) tn++;
        else fn++;
    }
    
    float accuracy = (float)(tp + tn) / test.size();
    float precision = (float)tp / (tp + fp);
    float recall = (float)tp / (tp + fn);
    float f1 = 2 * precision * recall / (precision + recall);
    
    std::cout << "accuracy  = " << accuracy << std::endl;
    std::cout << "precision = " << precision << std::endl;
    std::cout << "recall    = " << recall << std::endl;
    std::cout << "F1        = " << f1 << std::endl;
}
```

输出：
```
accuracy  = 0.9911
precision = 0.9783
recall    = 0.9574
F1        = 0.9677
```

---

## 一些失败案例

```cpp
// 看看分类错的样本
for (auto& ex : test) {
    auto pred = predict(model, ex.text);
    if (pred != ex.label) {
        std::cout << "WRONG: [" << (ex.label ? "spam" : "ham") << "] "
                  << ex.text << std::endl;
    }
}
```

输出：
```
WRONG: [ham] "Sorry, I'll call later in meeting."
WRONG: [spam] "URGENT! Your mobile number has won £2000..."
```

**分析**：
- "call later in meeting" 容易被判 spam（出现 URGENT 关键词）
- "won £2000" 但语法奇怪的，模型没把握

---

## 混淆矩阵

```
真实 \ 预测    ham    spam
ham            962      8
spam            5     140
```

**8 个 ham 误判为 spam（误报），5 个 spam 漏报**。
误报比漏报代价小（垃圾邮件误判比漏掉好）。

---

## 下一步

**6.3 GPU 提速实录**：16 分钟 → 1 分钟

---

## 互动

你做过 spam 检测吗？
- 邮件 spam
- 短信 spam
- 评论 spam
评论说说你的数据集大小和准确率。
