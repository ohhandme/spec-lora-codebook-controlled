# 云边教师码本指导 DraftLoRA

公开实验规范。原计划文件 SHA-256 为 54f2bb036885ea45b2528cd7af8193a838316ad4e6fa0caa9cfdae9ef329fc05。主基线使用原始 DraftLoRA；同一教师批验证器用于所有对照。

## 2. 主baseline B0：原样 DraftLoRA，必须不可变

- 固定0.6B基座、原未合并自定义LoRA，checkpoint epoch1.pt SHA b80519724c78cf494b2f29930370331a5231040e759f5ab28593090e02670b5b。
- 基座manifest SHA 3aa01f3c1f4b9bf488fbe696a374775e4ebf371be1a1836f5a1bda02256c3ee4。
- 原LoRA down/up，FP32，rank8、alpha16、scale2；28层q/v，56模块，112tensor、1146880参数，非PEFT、未merge。保留原dtype、cast、加法顺序。
- baseline不继续训练、不改原参数、不生成动态门控、不使用教师反馈、不加修复头。baseline最好直接运行没有新增rank hook的原前向；不能把“有一堆额外模块但给零”当作性能baseline。
- 新评估共用canonical Target批验/KV；改变的是共同验证执行入口，不是改baseline草稿模型。所有实验的Target参数/贪心接受规则相同。
- B0质量与成本均实际跑同一套请求。零门控/SHADOW仅是额外正确性对照，不替代B0。
- 分别记录原参数文件SHA、加载后的tensor digest、dtype、module schema；验证前后完全相同。禁止给B0恢复训练optimizer或覆盖原checkpoint。


## 3. 反馈参考 C0：冻结 Task78 实际架构，不擅自换接口

C0是新增方法的参考，不是用户的主baseline。任何方法都报告相对B0总体收益；只用与其直接父实验的差异归因具体机制。

Task78当前反馈架构必须准确沿用：
- 合法输入为Target H(4096)+拒绝/纠正词Draft embedding(各1024)+accept/8；全收有明确空拒绝标志。H来自生成correction/bonus的同次决策前向、纠正/bonus之前的前缀。
- encoder两层MLP，8组×256分类；每组code维16，共128维latent。codebook normal std=.02，receiver Linear(128,R,bias=False)零初始化，R由schema给出，当前448；g=.5*tanh(receiver(code))。
- 硬argmax前向、softmax直通梯度、温度1。训练仅更新encoder/codebook/receiver；基座、原A/B和Target冻结。
- 这128维是Task78实际latent配置，不是输出词表修复头；不能为迎合旧“无128强制”说明就悄悄改成直接448维。配置可支持其他latent，但仅在明确结构实验变化。
- code为64bit=8rawbytes，接收端查码+receiver得到rank gate，不恢复全H、不额外产生词表logits。
- 明确解耦draft_window、group_count、bits_per_index、payload_bytes；接受数归一化/全收判断不能再依赖WIRE_BYTES，16组4bit仍是8词窗口。

实现关闭动态模块的精确原分支；初始0贡献、数步后上游有有效梯度。零receiver导致最初encoder无梯度不算训练失败，但整个训练始终无梯度必须报错。

## 4. 固定数据、训练与选择合同

默认正式recipe沿Task78：train3986请求/156688窗/1217488监督tokens，calibration503/19736窗；固定5000索引与H绑定校验、家族隔离、原3986多窗历史重演。不得加载保存的旧Draft activations代替当前LoRA前向。

Task78的eval200是calibration503子集，只是开发诊断，不能叫独立测试；保留511不自动读取/评估。新增正式结果需要之后由执行席按授权提供真正留出合同。本轮只实现接口和禁止泄漏的检查，不擅自开放保留集。

默认5epoch、effective windows32、lr1e-4、WD.01、seed42，其他种子可配置；严格沿实际数据次序、micro累计、mask与按窗均值定义。每组的train seed/data order/有效监督token/总训练配额相同；不同loss不偷偷改变窗口权重。原5epoch不是当场自动运行指令。

- 核心正式选模统一沿Task78 calibration CE，并同时保存固定最后checkpoint结果；不能一组按CE、一组按接受数挑最好。若以后改变选择策略，登记独立实验因素。
- warm/pretrain/classifier bootstrap均占同一总训练配额，记录各阶段的监督token、更新、FLOPs/实测训练时间；可另报告同任务token与同计算预算两口径，不能假称二者自动相同。
- Kmeans/LBG只能拟合训练集，有限deterministic迭代、记录随机种子/空簇处理；不用开发/测试H拟合码字。
- 每条样本保留H的layer/prefix hash/absolute position/source observation key，标签只监督不进入反馈输入。
- 当前训练是RAW历史窗口上的offline teacher forcing，保持这个数据分布作固定因素；不得暗称on-policy闭环训练。可实现合法短展开，但自采样/重新采集H要另有数据合同，不在本轮生产自动执行。

## 5. 受控实验表与机器校验（arm_matrix.json是实现清单）

所有研究方法对B0报告总体增益。下列parent对照用于归因；同一表行改变一个语义因素。架构因素导致的参数/计算派生变化须报告，不伪称参数完全相同。

| ID | 直接parent | 唯一研究因素 |
|---|---|---|
| B0 | 无 | 原样DraftLoRA，不训练、不反馈 |
| C0 | B0（总体方法对照） | 加64bit反馈控制包；此比较只支持总体方法增益 |
| C_LOCAL | C0 | H置零，已知协议保留，encoder/码本/receiver/8B预算不变 |
| C_CONST | C0 | 全部条件固定，结构/预算保留，检容量/常数控制 |
| C_SHUFFLE | C0 | 仅H按同split可比边界跨请求打乱，协议不动 |
| T_SOFT_HARD | C0 | 前1/4配额soft混合，后3/4硬选码，其他相同 |
| T_STAGE_RAND | C0 | 分阶段任务控制预热+分类bootstrap+硬CE训练的整套schedule |
| T_STAGE_DATA | T_STAGE_RAND | 初始化原型由随机改为训练控制向量聚类 |
| L_MARGIN | C0 | CE增加Target词与最强竞争词margin辅助 |
| L_PREFIX | C0 | CE增加连续匹配的软前缀辅助，其他不动 |
| U_BALANCE | C0 | 增加跨样本码字使用率正则 |
| U_SHARP | C0 | 增加每样本分配明确性正则 |
| U_BALANCE_SHARP | U_BALANCE | 只再加sharp项；用此比较交互，不直接跳归因 |
| G16_K16 | C0 | 同64bit分组由8×256×16改16×16×8，latent仍128、receiver不变 |
| N_NEAREST | C0 | 分类式选码改为最近邻VQ家族（同bit/latent），仅作家族整体比较 |
| N_EMA | N_NEAREST | codebook更新器GD改EMA，损失/更新派生约束明确 |
| N_RVQ | N_NEAREST | 同64bit改两级分组残差结构，家族比较，报告容量差异 |
| D_FUSED | C0已训checkpoint | 推理解码实现融合，同一权重不重训 |
| M4_FIXED/M8_FIXED/M16_FIXED | 同一个最大预算母架构 | 训练固定预算4/8/16B，比较每预算专用模型 |
| M_MIX | M8_FIXED | 相同母架构与配额，训练预算分布改为4/8/16B混合 |
| M_MIX_E4/E8/E16 | 同M_MIX checkpoint | 仅评估预算不同，不能标成三次训练 |
| M_POLICY | M_MIX | 同一checkpoint，固定预算选择改轻量策略，选择成本计入 |
| J_H/J_LOCAL（默认不启用） | C0/C_LOCAL | 仅额外允许实验副本的原LoRA继续训练，配额一致，不能改B0 |

N_*、M_*、J_*是独立扩展配置，不纳入第一轮正式GPU启动。代码应真实支持并在tiny上覆盖；不是仅注册名字。若资源/库不支持，交付必须明确未实现，不以占位通过。

机器校验必须对完全展开的config/数据/模块/训练manifest比较：列出parent→child所有差异，核allowed factor及其派生字段；未声明的lr/seed/order/参数冻结/选择策略/精度/Target版本变化直接拒绝合并结果。命名、输出路径等非研究元数据单独排除。不同mask/bit预算/latent引出的形状变化也列出。

T_STAGE_RAND对C0测试的是整个分阶段程序，不可声称其中每一步单独有效；T_STAGE_DATA对T_STAGE_RAND才单独识别数据驱动原型初始化。

## 6. 损失与训练程序的具体定义

### 6.1 CE、margin与soft-prefix

沿formal78/objective.py复用当前LoRA激活与逐token历史g重演，输出合法预测位置的logits、labels和mask。每窗先按有效token数归一，再按原合同累计，不让长窗隐含增权。

CE = mean_valid[-log q_D(y_i^T)]。
margin m_i = z_D(y_i^T) - max_{v!=y_i^T} z_D(v)。
L_margin = mean_valid softplus((delta-m_i)/tau)。
s_i=sigmoid(m_i/tau)；A_soft=sum_{i=1..n}prod_{j<=i}s_j；L_prefix=-A_soft/n。
L_MARGIN=CE+lambda_m*L_margin；L_PREFIX=CE+lambda_p*L_prefix。lambda/tau/delta可配置，只在calibration调，不能混用不同最佳超参搜索预算。tiny默认值只是可复现测试值，不称正式推荐最优。稳定计算cumprod/logsigmoid，mask后的词不参与连乘；n=0拒绝样本，n=1/EOS有效。

以上是代理损失，不是无偏真实接受率，也不保证提升。严格比较greedy序列及P(A>=i)。不能把概率接受p/q或随机采样损失混进当前greedy合同。

### 6.2 分类选码的usage/sharpness

令r_ngk=softmax(logits/tau)，bar_r_gk为有效训练样本均值：
L_balance=mean_g KL(bar_r_g || Uniform(K))；L_sharp=mean_ng H(r_ng)。最小化balance鼓励总体覆盖，最小化sharp鼓励单样本明确，符号不得反。

主训练硬前向不变，正则从soft分配取梯度。总损失分别只加当前arm指定项。每组另统计hard argmax频率/perplexity/死码/最大占比与组间重合，不能只报soft entropy。正则对历史receipt同一窗口不重复计数，PAD不入统计。

固定8B时entropy下降不减少字节。无坍塌证据不承诺正则有益；不自动死码重置。重置如另实现，必须成为独立配置并记录被重置参数及optimizer状态。

### 6.3 两种日程

T_SOFT_HARD：同一结构、同一初始化、同N配额，前N/4使用soft码混合task CE，后3N/4必须实际硬前向；部署只hard编号。保存硬/软差距诊断，最终不得用soft评估冒充wire。

T_STAGE_*：建议配额1/4任务连续控制预热、1/4选码分类bootstrap、1/2硬码CE；比例可配置但预注册、两组相同。连续参考只作离线初始化，不部署全H接收器。
- 两组使用同一个按训练集任务目标得到的连续latent参考和相同receiver初始化；若参考不能提供有效信号，仍保留失败结果，不能用测试集换教师。
- RAND原型来自预注册随机规则；DATA从同一训练latent分组聚类；其余初始化/数据/配额完全同。
- bootstrap标签来自对应原型分配，拟合encoder分类输出。后续book/encoder/receiver仍用硬码task CE联合优化。
- 训练预算记录任务监督和prototype标签监督两类；不把bootstrap伪标签数当新增Target监督数。

### 6.4 最近邻/EMA/RVQ扩展

N_NEAREST显式定义encoder连续latent、分组距离和码字，标准ST、embedding/commitment损失，与分类选码C0区分。commitment不是直接套在分类logits与码字上。
N_EMA以训练assign count/vector的EMA更新book，禁止同时给book做optimizer和EMA两次更新；encoder/receiver仍任务梯度。当前EMA系数/epsilon写入合同，仅训练更新，eval冻结。
N_RVQ可用8组、每组两级K16、每级维16残差量化，8*2*4=64bit；重建仍128latent。与最近邻单级K256相比码本容量变化必须报告，不把它说成纯loss效应。编码串行两级，计入成本。

### 6.5 多码率母架构与策略

明确一个可序列化最大预算架构，所有M_*共用：8组×2级，每级最多256码、每组16维；两级向量相加为128latent，receiver相同。
- 4B：第一级每组只从前16个码选，8*4bit；第二级缺失。
- 8B：第一级每组256码，8*8bit；第二级缺失。
- 16B：两级每组256码，16*8bit。
- 两级可先采用并行分类的可加分量，不能冒称经典串行RVQ。其物理packet规定预算ID/级数、4bit顺序/有效长度；缺失级贡献0；bit packing不允许歧义。
- M*_FIXED和M_MIX相同母架构、初始化与总监督预算；只改变训练budget分布。M_MIX各预算任务损失/采样概率预注册，不用8B单独权重给16B额外训练。
- M_MIX_E*只加载同一个checkpoint，锁dtype/选择语义，硬wire比对。不能任意截现有8B当4B，也不能调用三个codec来挑一个。
- M_POLICY提供cheap lookup/阈值策略接口，用训练/开发的accept收益与codec/network成本拟合；不急RL、不在线反传。codebook每轮只编码/解码一次，策略输入仅当时可见本地状态/历史及网络信息。策略拟合数据与codebook评估留出分开，成本含策略本身。
- 反馈影响下一轮，当前轮贪心局部选择只是近似；最终以整请求验证，不声明闭式最优。

### 6.6 可选继续训练LoRA

默认所有正式核心arm冻结最佳LoRA。J_H/J_LOCAL仅在独立实验副本启用，不能写回原文件；同一adapter训练配额，保留J_LOCAL区分继续训练与教师内容贡献。若选模块需要requires_grad，升级RankGates的已知限制且仅该配置放开，不影响其他arm。代码可实现，CPU验证，生产运行另登记。

## 7. 推理、批验证与因果状态（共同固定因素）

Target用canonical BatchTarget：KV保持confirmed[:-1]，pending末词+候选整窗一次causal forward；每非初始化验证轮恰好1次Target decoder调用，row j的logits/hidden对应confirmed+accepted前缀。首拒绝或bonus取其同一行H；裁去拒绝后缀，校验各层cache长度。

- 记录forward实际输入长度/次数、cache before/raw/kept、hidden row/prefix hash、首拒绝、commit及terminal。不要只写断言成功而无实际计数hook。
- 原Target权重冻结；greedy沿native logits argmax，不加接受容差，不用修复logits替换Target判断。
- 初始prefill单列。terminal不再跑codec、额外Target forward或下一词更新；已经执行的批验证成本照计。
- Draft仍沿历史token-time门控：accepted前缀旧g；拒绝后correction新g；全收未消费末draft旧g、bonus新g；reset清空。完整参考重演各token历史g，不能全prefix套当前g。
- controller/解码每轮一次，下一批首候选真正使用新反馈；保持原A/B/scale和模型训练eval状态，forward保留codec训练梯度。
- 真实BF16可能与旧串行输出不同：核canonical批验自己的原生计算/因果/缓存不变量，记录漂移；旧oracle仅诊断。不要再把“必须等旧串行每位相同”变成额外冻结条件。
- FP32 tiny上可用独立causal模型计算参考做数学对账；BF16 tiny记录maxdiff/argmax差及真实分支，不能靠巨大容差掩盖接口错误。

## 8. 解码融合与时延接口

D_FUSED只对同一C0 checkpoint改变部署实现，不重新训练、不给其不同数据。当前有Linear(128,R)，不能预先逐码epsilon*tanh后求和：tanh不可跨求和交换。
可以预计算各组码字经receiver对应列块的贡献V_gk=W[:,slice_g]C_gk（有bias则只加一次），接收查表求和后再epsilon*tanh。数学上等价，但浮点加法次序改变；必须记录dtype/device下gate/logits/候选偏差。若接受轨迹改变，列为近似实现，不能标无损提速；保留原decode默认。
预计算表会增存储：G*K*R。G8/K256/R448时917504数，BF16约1.75MiB；原C0码表32768数加receiver57344参数，不可漏算内存或宣称表更小。
每轮decode缓存，别逐token重复；LoRA rank乘法每层每词仍有代价。encoder/quant/同步/pack/frame/transport/unpack/decode/设置g及token门控分别计数，nested计时不累加。应用帧字节与TCP/IP/实际RTT分开。

本轮实现CPU分项基准和可接网络入口；若未跑生产GPU，不写真实GPU速度。以后正式对比B0与各方法必须在同batchTarget/DraftKV入口，并另外给独立Target正常KV连续生成基准；不为Target每词人为加RTT。

## 9. 必须实际运行的CPU/tiny验证


1. B0无新增hook/feedback/optimizer，hidden/logits/候选同原LoRA；所有核心训练后base+adapter digest不变，checkpoint加载/恢复拒错绑定。
2. Target实际每轮1forward，输入长度整窗+pending；首拒/中拒/全收bonus/EOS/预算截断/请求reset/非法缓存；预测/commit/H行与独立FP32因果参考一致。不能用原串行代码充当batch路径。
3. 每个loss的真实端到端梯度和参数更新，mask/短窗/不等长窗/极值稳定性；构造“首词错后续全对”vs“首词对”的例子证明prefix代理区别于仅后词好看；不把导数非零当接受提升。
4. C0/train hard前向与真实pack-decode、保存加载一致；4/8/16B实际长度、nibble边界0/15/255、非法长度/版本/越界、None、NaN/Inf拒绝。同精度明确是否逐位一致，跨精度不冒称。
5. T_STAGE_RAND/DATA除原型规则外相同配额/参考/seed；聚类不读取validation/test；soft-to-hard最终真hard。N_EMA eval不更新统计、GD与EMA不双更新；RVQ真实两级作用。
6. usage统计hard/soft分开，mask不污染；balance/sharp符号通过人为分布检验。microbatch划分改变时，batch级regularizer要保持effective-batch语义，不能无意变成每micro独立计算；必要时先收集小分配张量再计算，不要求攒全模型激活。
7. 固定原权重的tiny闭环，B0/C0/LOCAL/SHUFFLE及所有新增loss/config至少走实际训练若干步后再实际批验证；并运行多轮历史g/KV对账。新增配置不能只生成JSON而未跑。
8. 实际本机TCP至少覆盖B0、C0、G16、M_MIX三预算和非法消息，候选/接受/commit/payload一致性。SHADOW诊断应不改静态候选/接受；SHADOW自身decode开销不能混入B0。
9. 机器控制变量校验能故意抓住未声明改seed/lr/LoRA/Target/数据/latent/预算/选择策略。不同arm恢复目录或parentcheckpoint错配应拒绝，已存在不同contract的out不覆盖。
10. 短CPU优化用于检查通路，不声称tiny最佳。分项成本若用生产shape随机tensor，明确不是生产质量或GPU收益。固定全流程原件、trace与失败样本。

如上述测试暴露真实缺陷，先修再复测；不能只重复实现逻辑的断言就称独立正确。某个可选扩展未完成要直说，不把核心已过写成全表已过。

## 10. 交付与验收

交付独立repo/commit或bundle及SHA、program副本、arm_matrix.json展开器、真实配置、train/evaluate/benchmark命令、CPU tiny全部执行报告与逐例JSON、源码清单和asset/contract绑定。

至少命令能力：list-arms、validate-plan、train --arm --device --out、evaluate --arm --checkpoint --roster、cpu-smoke、benchmark-codec；名称可按repo习惯调整，但必须可直接复现，训练不能只会随机夹具而没有正式data adapter。

汇总每arm parent/唯一因素/派生参数和计算、有效监督token/更新/阶段预算、可训练参数名、wire byte、acceptsum/roundsum、P(A>=i)、A=0比率、按请求配对差及区间。保留last checkpoint与统一CE选模两口径，不cherry-pick；报告dev200非独立测试。

