# CausalTwinSim: Phân tích toàn bộ các module và flow hoạt động

Dưới đây là flow đầy đủ của **CausalTwinSim**, phân tích theo từng module: nhận dữ liệu gì, xử lý gì, tạo output gì, tại sao cần, và output đó được chuyển sang module tiếp theo như thế nào.

Tôi dùng xuyên suốt một ví dụ:

> **User U001** đã xem giày chạy bộ 5 lần, từng thêm vào giỏ rồi bỏ, thường mở email và trước đây có dùng coupon.  
> **Campaign C11**: “Giảm 15% giày chạy bộ trong 24 giờ, gửi qua email cho khách đã xem nhưng chưa mua.”

---

# 1. Tổng quan flow

```text
User history ──> User History Encoder ──> User state z
                                                │
Campaign text ─> Campaign Encoder ───────> Campaign vector e
                                                │
                                Semantic Intervention Adapter
                                                │
                                      Modified user dynamics
                                                │
                           Treatment-conditioned World Model
                                                │
                      ┌─────────────────────────┴─────────────────────────┐
                      │                                                   │
             Treatment twin                                      Control twin
             Có campaign                                         Không campaign
                      │                                                   │
             Treatment trajectories                              Control trajectories
                      └─────────────────────────┬─────────────────────────┘
                                                │
                                      Causal uplift estimation
                                                │
                                    Uncertainty and abstention
                                                │
                     Uplift map + funnel + deploy/pilot/reject
```

Có thể chia thành:

1. Data preparation  
2. User History Encoder  
3. Campaign Encoder  
4. Semantic Intervention Adapter  
5. Treatment-Conditioned World Model  
6. Causal Training Module  
7. Counterfactual Twin Rollout  
8. Uncertainty and Decision Module  

---

# 2. Module 0 — Data Preparation

Đây chưa phải neural-network module, nhưng rất quan trọng.

## Input

Dữ liệu lịch sử dạng event log:

| user_id | timestamp | event | product | category | price | campaign | treatment |
|---|---|---|---|---|---:|---|---:|
| U001 | Day -10 | view | Shoe A | running | 900000 | null | null |
| U001 | Day -8 | add_cart | Shoe A | running | 900000 | null | null |
| U001 | Day -8 | remove_cart | Shoe A | running | 900000 | null | null |
| U001 | Day -3 | open_email | null | null | 0 | C08 | 1 |
| U001 | Day 0 | assigned | null | null | 0 | C11 | 1 |
| U001 | Day 0 | open_email | null | null | 0 | C11 | 1 |
| U001 | Day 0 | purchase | Shoe A | running | 765000 | C11 | 1 |

Ngoài ra cần bảng campaign:

| campaign_id | description | discount | duration | channel | target |
|---|---|---:|---:|---|---|
| C11 | 15% off running shoes for 24 hours | 0.15 | 24h | email | viewed-not-purchased |

Và bảng assignment:

| user_id | campaign_id | assigned_treatment | assignment_probability |
|---|---|---:|---:|
| U001 | C11 | 1 | 0.5 |
| U002 | C11 | 0 | 0.5 |

## Xử lý

Dữ liệu được chia thành:

- **Pre-treatment history**: hành vi trước khi campaign bắt đầu.
- **Post-treatment trajectory**: hành vi sau khi campaign bắt đầu.
- **Outcome**: purchase, revenue, unsubscribe hoặc retention.

Ví dụ:

```text
Pre-treatment:
view → view → add_cart → remove_cart → open_email

Post-treatment:
open_email → view → apply_coupon → purchase
```

## Output

```text
User sequence H_i
Campaign C
Treatment indicator T
Observed post-treatment sequence
Observed outcome Y
```

Ví dụ dưới dạng object:

```python
{
    "user_id": "U001",
    "history": [
        "view_running_shoe",
        "view_running_shoe",
        "add_cart",
        "remove_cart",
        "open_previous_email"
    ],
    "campaign": "C11",
    "treatment": 1,
    "outcome": {
        "purchase": 1,
        "revenue": 765000
    }
}
```

---

# 3. Module 1 — User History Encoder

## Mục tiêu

Biến toàn bộ lịch sử của user thành một vector trạng thái hiện tại:

$$
z_{i,t}=F_{\text{user}}(H_{i,1:t},X_i)
$$

Trong đó $z_{i,t}$ là latent user state.

## Input

Lịch sử:

```text
View running shoes
→ View running shoes
→ Add to cart
→ Remove from cart
→ Open previous promotional email
```

Mỗi event có thể chứa:

```text
event_type
product/category
price
timestamp
device
channel
time_since_previous_event
```

## Cách xử lý

Mỗi event được mã hóa thành vector:

$$x_t = E_{\text{event}} + E_{\text{product}} + E_{\text{time}} + E_{\text{channel}}$$

Sau đó đưa cả chuỗi qua:

- Transformer;
- GRU/LSTM;
- hoặc state-space model.

Ví dụ:

```text
Event vectors
      ↓
Transformer
      ↓
User state z_U001
```

## Output

Về máy học, output là vector:

$$
z_{U001}\in\mathbb{R}^{d}
$$

Ví dụ:

```text
z_U001 = [0.72, -0.18, 0.55, ..., 0.31]
```

Vector này không nhất thiết có từng chiều mang nghĩa trực tiếp. Nhưng khi phân tích, có thể hiểu trạng thái đang mã hóa các yếu tố như:

```text
Product interest:         cao
Purchase intention:       khá cao
Price sensitivity:        trung bình-cao
Email responsiveness:     cao
Cart-abandonment tendency: cao
```

## Kiểu output

```text
Tensor/vector: [user_state_dimension]
```

Ví dụ:

```python
user_state.shape = [128]
```

## Tại sao cần?

Nếu chỉ nhìn event cuối:

```text
remove_cart
```

model không biết user:

- đã xem sản phẩm 5 lần hay chỉ một lần;
- thường dùng coupon hay không;
- có thường mở email không;
- đã mua sản phẩm tương tự chưa.

Module này giúp model nhớ toàn bộ lịch sử thay vì chỉ dùng:

$$
P(a_{t+1}\mid a_t)
$$

như transition model đơn giản.

## Nếu bỏ module này

Hai user cùng đang ở `view_product` có thể bị dự đoán giống nhau dù lịch sử hoàn toàn khác nhau.

---

# 4. Module 2 — Semantic Campaign Encoder

## Mục tiêu

Biến campaign mới thành một vector mô tả ý nghĩa và thuộc tính của campaign:

$$
e_C=F_{\text{campaign}}(\text{text},\text{attributes})
$$

## Input

### Text

```text
“Get 15% off running shoes for the next 24 hours.
Available to customers who viewed but did not purchase.”
```

### Structured attributes

```text
discount_type = percentage
discount_value = 0.15
duration_hours = 24
channel = email
category = running_shoes
target = viewed_not_purchased
minimum_spend = 0
```

## Cách xử lý

### Text branch

Text được đưa qua:

- Sentence-BERT;
- frozen language encoder;
- hoặc fine-tuned text encoder.

Tạo:

$$
e_{\text{text}}
$$

### Structured branch

Các thuộc tính category được embedding, giá trị số được chuẩn hóa:

$$e_{\text{structured}} = \mathrm{MLP}(\text{discount}, \text{duration}, \text{channel}, \text{category})$$

Sau đó kết hợp:

$$e_C = W[e_{\text{text}}; e_{\text{structured}}]$$

## Output

```text
Campaign semantic vector e_C
```

Ví dụ:

```python
campaign_embedding.shape = [64]
```

Có thể giải thích output ở mức khái niệm:

```text
Promotion type:       percentage discount
Strength:             medium
Urgency:              high
Product specificity:  running shoes
Channel:              email
Targeting:            cart/view abandoners
```

## Tại sao cần?

Campaign mới chưa từng xuất hiện sẽ không có learned campaign ID.

Ví dụ model chưa thấy C11, nhưng đã thấy:

```text
C03: 10% off sports shoes
C05: 20% flash sale
C08: email coupon for cart abandoners
```

Nhờ semantic encoder, model biết C11 có liên quan tới cả ba campaign trên.

## Nếu chỉ dùng campaign ID

```text
C11 → unknown token
```

Model không thể generalize sang campaign mới.

---

# 5. Module 3 — Semantic Intervention Adapter

Đây là module biến campaign từ một vector mô tả thành một **tác động lên cơ chế hành vi**.

## Câu hỏi module này trả lời

> Campaign này sẽ thay đổi phần nào trong behavior model?

Ví dụ campaign giảm giá không nên thay đổi mọi thứ. Nó có thể thay đổi:

- xác suất mở email;
- xác suất quay lại sản phẩm;
- xác suất áp dụng coupon;
- xác suất mua của user nhạy cảm với giá.

Nhưng không nên thay đổi mạnh:

- giới tính;
- sở thích màu sắc;
- thiết bị user thường sử dụng.

## Input

```text
User state z_U001
Campaign vector e_C11
```

## Cách xử lý đơn giản

Nối hai vector:

$$
h=[z_i;e_C]
$$

Nhưng cách này tương đối yếu vì campaign chỉ trở thành feature bổ sung.

## Cách xử lý đề xuất

Campaign tạo một adapter hoặc gate điều chỉnh world model:

$$
\Delta\theta_C=H_\psi(e_C)
$$

$$
\theta_C=\theta_0+\Delta\theta_C
$$

Hoặc dùng feature-wise modulation:

$$
\gamma_C,\beta_C=H_\psi(e_C)
$$

$$\tilde{z}_i=\gamma_C\odot z_i+\beta_C$$

Hiểu dễ dàng:

```text
Base user state
      +
Campaign effects
      ↓
Campaign-adjusted user state
```

## Ví dụ output

Đây là số minh họa:

```text
Email-open gate:            +0.18
Product-revisit gate:       +0.14
Coupon-use gate:            +0.31
Purchase-intention gate:    +0.09
Unsubscribe-risk gate:      +0.02
```

Hoặc machine output:

```python
adapter_parameters = {
    "gamma": tensor([..., ...]),
    "beta": tensor([..., ...])
}
```

## Output

Có hai dạng khả thi:

### Dạng A: adjusted state

$$\tilde{z}_{i,C}$$

```python
adjusted_user_state.shape = [128]
```

### Dạng B: campaign-specific model parameters

$$
\theta_C
$$

Ví dụ low-rank adapter matrices:

```python
adapter_A.shape = [128, 8]
adapter_B.shape = [8, 128]
```

## Tại sao cần?

Campaign encoder chỉ nói:

> Campaign này là giảm giá 15%, urgency cao, gửi qua email.

Intervention adapter chuyển thông tin đó thành:

> Với behavior model, hãy tăng sensitivity của price-sensitive users và email-responsive users.

## Điểm khác CXSimulator

CXSimulator dùng campaign embedding để dự đoán các cạnh quanh campaign node.

Module này dùng campaign embedding để **điều chỉnh toàn bộ dynamics có điều kiện theo user state**.

---

# 6. Module 4 — Treatment-Conditioned World Model

Đây là model trung tâm.

## Mục tiêu

Dự đoán điều gì xảy ra tiếp theo sau trạng thái hiện tại.

$$
P(a_{t+1}\mid z_t,e_C)
$$

Có thể mở rộng thành:

$$
P(a_{t+1},\Delta t_{t+1},r_{t+1}\mid z_t,e_C)
$$

Trong đó:

- $a_{t+1}$: action tiếp theo;
- $\Delta t_{t+1}$: thời gian tới action;
- $r_{t+1}$: reward/revenue;
- $z_{t+1}$: trạng thái mới.

## Input

Tại một bước:

```text
Current user state:
- đã mở email
- quan tâm cao tới giày chạy bộ
- từng bỏ giỏ
- khá nhạy cảm với discount

Campaign:
- giảm 15%
- thời hạn 24 giờ
```

Machine input:

```text
adjusted state z̃_t
campaign embedding e_C
current action/context
```

## Output 1: Next-action distribution

Ví dụ:

| Action tiếp theo | Probability |
|---|---:|
| view_product | 0.34 |
| apply_coupon | 0.18 |
| add_to_cart | 0.21 |
| purchase | 0.06 |
| ignore | 0.08 |
| exit | 0.13 |

Tổng bằng 1:

$$
\sum_a P(a_{t+1}=a)=1
$$

Kiểu dữ liệu:

```python
action_probs.shape = [number_of_actions]
```

## Output 2: Time-to-next-event

Ví dụ:

```text
Expected time to next event = 4.7 minutes
```

Hoặc phân phối:

```text
0–5 minutes:    0.55
5–30 minutes:   0.25
30 min–24 hrs:  0.15
No action:      0.05
```

## Output 3: Revenue/outcome distribution

Ví dụ:

```text
P(no purchase)     = 0.88
P(purchase £/₫...) = 0.12
Expected revenue   = 91,800 VND
```

## Output 4: Updated user state

Nếu sample action là `view_product`:

$$z_{t+1}=f(z_t,\text{view\_product},e_C)$$

User state có thể thay đổi:

```text
Product interest tăng
Time remaining giảm
Email state chuyển từ opened sang engaged
```

## Tại sao cần world model?

Một uplift classifier chỉ trả:

```text
P(purchase with campaign) = 12%
```

World model cho biết **campaign tác động qua đường nào**:

```text
open_email
→ view_product
→ apply_coupon
→ add_cart
→ purchase
```

Nó giúp phân tích:

- campaign tăng attention nhưng không tăng purchase;
- user mắc lại ở bước add-to-cart;
- discount hiệu quả nhưng checkout friction vẫn lớn;
- campaign gây unsubscribe sau nhiều lần exposure.

---

# 7. Module 5 — Causal Training Module

Module này chủ yếu hoạt động khi training, không phải inference.

## Vấn đề

Giả sử doanh nghiệp gửi discount chủ yếu cho user đã gần mua.

Dữ liệu cho thấy:

```text
Treatment conversion = 20%
Control conversion   = 8%
```

Không thể ngay lập tức kết luận uplift là 12%.

Có thể treatment users vốn đã có xác suất mua 17%, còn campaign chỉ tăng thêm 3%.

## Input

- User history $H_i$
- Treatment assignment $T_i$
- Assignment probability $p_i$
- Observed outcome $Y_i$
- Campaign $C_i$

## Trường hợp A: Randomized A/B data

Ví dụ:

$$T_i \sim \mathrm{Bernoulli}(0.5)$$

Khi đó treatment và control được random, việc so sánh đáng tin cậy hơn.

## Trường hợp B: Observational data

Học propensity score:

$$
\hat e_i=P(T_i=1\mid H_i,X_i)
$$

Ví dụ:

```text
U001 propensity of receiving campaign = 0.82
U002 propensity of receiving campaign = 0.15
```

Sau đó dùng:

- inverse propensity weighting;
- doubly robust loss;
- overlap filtering.

## Doubly robust target

$$\tilde Y_i(1) = \hat\mu_1(H_i) + \frac{T_i}{\hat e_i}\left[Y_i-\hat\mu_1(H_i)\right]$$

Tương tự cho control.

## Output khi training

Module này không tạo final report. Nó tạo:

```text
Propensity model
Treatment outcome model
Control outcome model
Causally adjusted training targets
Causal loss
```

Ví dụ:

```text
Observed treatment outcome:       1
Predicted treatment outcome:      0.18
Propensity score:                  0.82
Doubly robust adjusted target:     0.21
```

## Output cuối cùng của module

Các tham số world model được cập nhật sao cho model học:

> Campaign tạo ra thay đổi gì?

thay vì chỉ học:

> Người nhận campaign thường có hành vi gì?

## Nếu không có module này

Model có thể học nhầm targeting policy thành campaign effect.

---

# 8. Module 6 — Counterfactual Twin Rollout

Đây là phần inference quan trọng nhất.

## Input

```text
Initial user state z_U001
New campaign C11
No-campaign condition C0
Trained world model
```

## Tạo hai twins

### Treatment twin

$$
z_0^{(1)}=z_{U001}
$$

Campaign:

$$
C=C11
$$

### Control twin

$$
z_0^{(0)}=z_{U001}
$$

Campaign:

$$
C=C0
$$

Hai twins bắt đầu từ **cùng một user state**.

---

## Một treatment rollout

### Bước 1

```text
Current: campaign received
```

Model output:

```text
open_email: 0.62
ignore:     0.38
```

Sample:

```text
open_email
```

### Bước 2

```text
Current: email opened
```

Model output:

```text
view_product: 0.48
exit:         0.32
browse_other: 0.20
```

Sample:

```text
view_product
```

### Bước 3

```text
Current: product viewed
```

Model output:

```text
apply_coupon: 0.25
add_cart:     0.24
exit:         0.41
compare:      0.10
```

Sample:

```text
apply_coupon
```

Cuối cùng:

```text
receive_campaign
→ open_email
→ view_product
→ apply_coupon
→ add_cart
→ purchase
```

Đây là một treatment trajectory.

---

## Một control rollout

Không có campaign:

```text
no_campaign
→ natural_visit
→ view_product
→ exit
```

---

## Không chỉ chạy một lần

Chạy $K=1000$ hoặc $K=10000$ lần.

### Treatment outputs

```text
1,000 treatment trajectories
230 kết thúc bằng purchase
```

$$
\hat P(Y=1\mid C11)=0.23
$$

### Control outputs

```text
1,000 control trajectories
130 kết thúc bằng purchase
```

$$
\hat P(Y=1\mid C0)=0.13
$$

## Output

### Trajectory sets

```python
treatment_trajectories: list[list[event]]
control_trajectories: list[list[event]]
```

### Outcome estimates

```text
Treatment purchase probability = 0.23
Control purchase probability   = 0.13
```

### Individual uplift

$$
\hat\tau_{U001}=0.23-0.13=0.10
$$

### Revenue uplift

Ví dụ:

```text
Expected treatment revenue = 176,000 VND
Expected control revenue   = 94,000 VND
Incremental revenue        = 82,000 VND
```

## Common random numbers

Treatment và control nên dùng cùng random noise.

Ví dụ cùng số ngẫu nhiên $u_1=0.4$:

- Treatment probabilities khiến 0.4 rơi vào `open_email`;
- Control probabilities có thể khiến 0.4 rơi vào `natural_visit`.

Việc này làm chênh lệch ổn định hơn.

Nhưng cần nhớ:

> Shared noise giảm variance, không tự tạo causal validity.

---

# 9. Module 7 — Uncertainty and Abstention

## Mục tiêu

Xác định model có đủ chắc chắn để tin kết quả hay không.

## Input

```text
Treatment rollouts
Control rollouts
Predictions từ nhiều models/ensemble
Campaign embedding
Training campaign embeddings
Propensity/overlap statistics
```

## Nguồn uncertainty 1: Rollout variance

Ví dụ 5 batch rollout cho uplift:

```text
4.2%, 7.1%, 5.8%, 6.4%, 3.9%
```

Variance khá cao.

## Nguồn uncertainty 2: Model uncertainty

Năm models dự đoán:

```text
Model 1: +6.0%
Model 2: +5.4%
Model 3: +7.2%
Model 4: +1.1%
Model 5: +6.5%
```

Model 4 bất đồng mạnh.

## Nguồn uncertainty 3: Campaign OOD

Campaign training:

```text
discount
free shipping
coupon
urgency email
```

Campaign mới:

```text
“Buy a product to receive an NFT and governance voting rights”
```

Embedding quá xa training campaigns.

## Nguồn uncertainty 4: Lack of overlap

Ví dụ gần như tất cả high-intent users đều nhận treatment:

```text
P(T=1 | high intent) = 0.98
```

Không có đủ control users tương tự để xác định effect.

## Output

### Confidence interval

```text
Predicted uplift = +6.0%
90% interval = [+3.1%, +8.7%]
```

### OOD score

```text
Campaign OOD score = 0.14
```

### Overlap score

```text
Treatment-control overlap = good
```

### Decision

```text
DEPLOY
```

Hoặc:

```text
Predicted uplift = +3.0%
90% interval = [-4.0%, +9.0%]
Campaign OOD score = 0.78
Decision = PILOT_REQUIRED
```

## Quy tắc minh họa

```text
Nếu lower bound > 0 và OOD thấp:
    DEPLOY

Nếu upper bound < 0:
    REJECT

Nếu interval chứa 0 hoặc OOD cao:
    PILOT_REQUIRED
```

---

# 10. Module 8 — Reporting Layer

Đây không nhất thiết là một learning module, mà là cách biểu diễn kết quả.

## Output 1: Top trajectories

### Treatment

| Trajectory | Tỷ lệ |
|---|---:|
| Open → View → Cart → Purchase | 23% |
| Open → View → Exit | 19% |
| Ignore campaign | 38% |
| Open → View → Cart → Exit | 20% |

### Control

| Trajectory | Tỷ lệ |
|---|---:|
| Natural visit → View → Purchase | 13% |
| Natural visit → View → Exit | 18% |
| No activity | 69% |

---

## Output 2: Counterfactual funnel

| Funnel stage | Treatment | Control | Increment |
|---|---:|---:|---:|
| Open/visit | 62% | 31% | +31% |
| View product | 48% | 24% | +24% |
| Add to cart | 29% | 17% | +12% |
| Purchase | 23% | 13% | +10% |

Điều này cho biết campaign tác động mạnh nhất ở bước nào.

---

## Output 3: Uplift map

| Segment | Treatment CVR | Control CVR | Uplift | Interval | Decision |
|---|---:|---:|---:|---:|---|
| Cart abandoners | 24% | 14% | +10% | [6.5%, 13.1%] | Deploy |
| Repeat viewers | 13% | 8% | +5% | [2.1%, 7.7%] | Deploy |
| One-time viewers | 3.0% | 2.2% | +0.8% | [-1.1%, 2.6%] | Pilot |
| Email-fatigue users | 1.4% | 1.8% | -0.4% | [-1.0%, -0.1%] | Reject |

---

## Output 4: Campaign ranking

| Campaign | Uplift | Revenue uplift | Uncertainty | Recommendation |
|---|---:|---:|---|---|
| Free shipping | +8.3% | 510M | Low | Deploy |
| 15% discount | +6.1% | 420M | Low | Deploy |
| Coupon 100k | +3.0% | 170M | High | Pilot |

---

# 11. Training flow hoàn chỉnh

## Bước 1: Xây user histories

```text
Raw logs
→ sorted event sequences
→ pre-treatment histories
→ post-treatment targets
```

## Bước 2: Học user encoder

Học representation của hành vi trước campaign.

```text
History → z_user
```

## Bước 3: Học campaign encoder

```text
Campaign text + metadata → e_campaign
```

## Bước 4: Học base world model

Ban đầu học behavior nói chung:

$$
P(a_{t+1}\mid z_t)
$$

## Bước 5: Học intervention adapter

Học campaign làm thay đổi dynamics như thế nào:

$$
P(a_{t+1}\mid z_t,e_C)
$$

## Bước 6: Causal adjustment

Dùng randomized assignment hoặc propensity/DR loss để tách campaign effect khỏi targeting bias.

## Bước 7: Campaign-level validation

Split:

```text
Train campaigns:      C1–C15
Validation campaigns: C16–C18
Test campaigns:       C19–C20
```

Không được random split user của tất cả campaigns, vì như vậy model đã thấy campaign test.

## Bước 8: Uncertainty calibration

Dùng prediction error trên validation campaigns để học confidence interval và abstention threshold.

---

# 12. Inference flow khi có một campaign hoàn toàn mới

Campaign C21:

> “Free shipping for running-shoe orders above 600,000 VND, valid for 48 hours.”

## Bước 1

Campaign encoder:

```text
C21 → e_C21
```

## Bước 2

Với từng user:

```text
history → z_user
```

## Bước 3

Adapter:

```text
z_user + e_C21
→ campaign-adjusted dynamics
```

## Bước 4

Treatment rollout:

```text
Run 1,000 trajectories with C21
```

## Bước 5

Control rollout:

```text
Run 1,000 trajectories with no campaign
```

## Bước 6

Tính:

```text
Purchase uplift
Revenue uplift
Unsubscribe risk
Changes in funnel stages
```

## Bước 7

Tính uncertainty:

```text
Confidence interval
OOD score
Overlap quality
```

## Bước 8

Tổng hợp:

```text
User-level uplift
→ segment-level uplift map
→ campaign recommendation
```

---

# 13. Ví dụ output cuối cùng

```text
Campaign:
Free shipping for running shoes above 600,000 VND

Target population:
100,000 users

Predicted treatment conversion:
11.8%

Predicted control conversion:
7.1%

Predicted uplift:
+4.7 percentage points

90% confidence interval:
[+2.8%, +6.4%]

Expected incremental purchases:
4,700

Expected incremental revenue:
3.1 billion VND

Best segment:
Users who abandoned cart because of shipping cost

Negative-risk segment:
Users with high promotional email fatigue

OOD score:
Low

Final decision:
Deploy to high-uplift segments
```

---

# 14. Vai trò chính của từng module

| Module | Câu hỏi nó trả lời | Output |
|---|---|---|
| User History Encoder | User hiện đang ở trạng thái nào? | $z_i$ |
| Campaign Encoder | Campaign mới có ý nghĩa gì? | $e_C$ |
| Intervention Adapter | Campaign thay đổi behavior dynamics như thế nào? | $\tilde{z}$ hoặc $\Delta\theta_C$ |
| World Model | Hành động tiếp theo là gì? | Action probabilities, time, revenue, next state |
| Causal Module | Thay đổi đó có thật sự do campaign không? | Adjusted targets/losses |
| Twin Rollout | Có và không có campaign khác nhau thế nào? | Treatment/control trajectories |
| Uncertainty Module | Kết quả có đáng tin không? | Interval, OOD, abstention |
| Reporting Layer | Marketer nên làm gì? | Uplift map, funnel, ranking, decision |

Cách nhớ ngắn nhất:

```text
User Encoder:
“Người này hiện là ai?”

Campaign Encoder:
“Campaign này là gì?”

Adapter:
“Campaign tác động lên người này ở điểm nào?”

World Model:
“Họ có thể làm gì tiếp theo?”

Causal Module:
“Thay đổi có thật sự do campaign không?”

Twin Rollout:
“Nếu có và không có campaign thì khác nhau thế nào?”

Uncertainty:
“Ta có đủ chắc chắn để tin kết quả không?”

Reporting:
“Nên deploy, reject hay chạy pilot?”
```
