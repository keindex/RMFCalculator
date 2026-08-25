---
name: add-latex-formulas
description: '为 Python 物理计算代码添加 LaTeX 格式的数学公式注释和中文说明。当用户要求添加公式注释、解释物理代码、或为代码添加 LaTeX 文档时触发。Use when: adding LaTeX formula comments, explaining physics code, documenting mathematical expressions in Python.'
argument-hint: '要添加 LaTeX 公式注释的 Python 文件或函数'
---

# 添加 LaTeX 公式注释技能

为 Python 物理计算代码添加中文注释和 LaTeX 格式的数学公式解释。

## 何时使用

- 用户要求为代码添加 LaTeX 格式的公式注释
- 需要解释物理计算代码中的数学表达式
- 为函数添加带 LaTeX 公式的 docstring
- 为 RMF 模型、核物质状态方程等物理代码添加文档

## 工作流程

### 1. 读取代码文件

使用 `file_search` 和 `read_file` 读取目标 Python 文件，理解代码结构和物理含义。

### 2. 为函数添加 docstring

每个函数添加中文 docstring，包含：
- 函数功能描述
- 核心物理公式（LaTeX 格式）
- 参数说明（含单位和物理意义）
- 返回值说明

### 3. 为关键计算添加行内注释

为以下内容添加带 LaTeX 公式的行内注释：
- 物理量定义（如密度、化学势、有效质量）
- 场方程求解
- 能量密度和压强计算
- 循环和条件判断的物理来源

## 注释规范

### LaTeX 希腊字母

所有希腊字母必须使用 LaTeX 格式：

| 字母 | LaTeX | 常见用途 |
|------|-------|---------|
| σ | `\sigma` | σ 介子场 |
| ω | `\omega` | ω 介子场 |
| ρ | `\rho` | ρ 介子场 / 数密度 |
| δ | `\delta` | δ 介子场 |
| μ | `\mu` | 化学势 |
| κ | `\kappa` | 非线性 σ 耦合 |
| λ | `\lambda` | 三次非线性耦合 |
| ζ | `\zeta` | ω 四次耦合 |
| Λ | `\Lambda` | 交叉耦合 |
| ε | `\epsilon` | 能量密度 |
| π | `\pi` | 圆周率 |

### 行内公式注释

```python
# 质子数密度: $\rho_p = \alpha \rho$, 其中 $\alpha$ 是质子分数
rho_p = alpha * rho
# 质子费米动量: $k_F^p = (3\pi^2 \rho_p)^{1/3}$, 单位: fm$^{-1}$
kf_p = (rho_p * (3 * math.pi**2))**(1/3)
```

### 函数 Docstring 模板

```python
def compute_INEOS_delta(rho, theta, initial_values=None):
    r"""
    计算无限核物质在给定密度下的介子场和化学势。
    
    求解 RMF 模型的介子场方程 (论文式 4.8-4.11):
    
    .. math::
        \begin{aligned}
        m_\sigma^2\bar\sigma &= g_\sigma\left[n^s - b_\sigma M(g_\sigma\bar\sigma)^2
          - c_\sigma(g_\sigma\bar\sigma)^3\right] \\
        m_\omega^2\bar\omega &= g_\omega\left[n - c_\omega(g_\omega\bar\omega)^3\right]
        \end{aligned}
    
    核子有效质量:
    
    .. math::
        M_J^* = M - g_\sigma\bar\sigma - g_\delta\bar\delta\,\tau_3^J
    
    Args:
        rho (float): 重子数密度, 单位: fm$^{-3}$
        theta (array): RMF 参数向量, 长度 10 或 13
        initial_values (array, optional): 初值 [$\sigma$, $\omega$, $\rho_{03}$, $\delta$, $\mu_n$, $\mu_p$]
    
    Returns:
        array: 求解结果 [$\sigma$, $\omega$, $\rho_{03}$, $\delta$, $\mu_n$, $\mu_p$]
    """
```

### 物理来源说明

对于循环和条件判断，说明其物理来源：

```python
# 遍历质子 (i=0) 和中子 (i=1)
# 核子有效质量: $M_i^* = M - g_\sigma\sigma - g_\delta\delta \cdot \tau_3^i$
# 其中 $\tau_3^p = +1$, $\tau_3^n = -1$
for i in range(2):
    m_eff_i = M - g_sigma * sigma - g_delta * delta * tau3[i]
```

## 项目特定知识

### RMFCalculator 项目约定

- 参数向量 `theta` 可为 10 元（无 δ）或 13 元（含 δ）
- 10 元: `[m_σ, m_ω, m_ρ, g_σ, g_ω, g_ρ, κ, λ₀, ζ, Λ_ω]`
- 13 元: `[m_σ, m_ω, m_ρ, m_δ, g_σ, g_ω, g_ρ, g_δ, κ, λ₀, ζ, Λ_ω, Λ_σδ]`
- 采用"大 g_ρ"约定: ρ 场方程源项为 $\tfrac12(\rho_p - \rho_n)$
- δ 耦合使用 τ₃ = ±1（不是 I₃ = ±1/2）
- 能量密度和压强单位: fm⁻⁴（自然单位），需乘以 $\hbar c = 197.327$ 转换为 MeV/fm³
- 模块级 docstring 含 LaTeX 反斜杠必须加 `r` 前缀

### 核心物理公式

**介子场方程**（论文式 4.8-4.11）:

$$
\begin{aligned}
m_\sigma^2\bar\sigma &= g_\sigma\left[n^s - b_\sigma M(g_\sigma\bar\sigma)^2 - c_\sigma(g_\sigma\bar\sigma)^3 + \Lambda_{\sigma\delta}(g_\sigma\bar\sigma)(g_\delta\bar\delta)^2\right] \\
m_\omega^2\bar\omega &= g_\omega\left[n - c_\omega(g_\omega\bar\omega)^3 - \lambda_v(g_\omega\bar\omega)(g_\rho\bar\rho)^2\right] \\
m_\rho^2\bar\rho &= g_\rho\left[(n_p - n_n) - \lambda_v(g_\rho\bar\rho)(g_\omega\bar\omega)^2\right] \\
m_\delta^2\bar\delta &= g_\delta\left[(n_p^s - n_n^s) + \Lambda_{\sigma\delta}(g_\delta\bar\delta)(g_\sigma\bar\sigma)^2\right]
\end{aligned}
$$

**核子有效质量**:

$$M_J^* = M - g_\sigma\bar\sigma - g_\delta\bar\delta\,\tau_3^J, \qquad J = p, n$$

**标量密度**:

$$n_J^s = \frac{1}{\pi^2}\int_0^{k_F^J} \frac{M_J^*}{\sqrt{k^2 + M_J^{*2}}} k^2 dk$$

**能量密度**:

$$
\mathcal{E} = \sum_J \mathcal{E}_J^{\text{kin}} + \frac12 m_\sigma^2\bar\sigma^2 + \frac12 m_\omega^2\bar\omega^2 + \frac12 m_\rho^2\bar\rho^2 + \frac12 m_\delta^2\bar\delta^2 + \mathcal{E}^{\text{NL}}
$$

**压强**:

$$P = \sum_J \mu_J n_J - \mathcal{E}$$

## 处理策略

### 注释处理
- **替换重写**: 用带 LaTeX 公式的新注释替换现有注释，确保公式完整准确
- 保留有意义的变量名和代码结构不变

### 处理模式
- **批量处理**: 一次处理整个模块/文件夹，保持注释风格一致
- 按文件依赖顺序处理（先底层工具函数，再高层业务函数）

### Docstring 详细程度
- **完整版**: 每个函数包含：
  - 功能描述（中文）
  - 核心物理公式（LaTeX `.. math::` 块）
  - 所有参数说明（含单位）
  - 返回值说明
  - 参考文献引用（如适用）

## 实施步骤

1. **读取文件** — 使用 `read_file` 读取目标 Python 文件
2. **识别函数** — 找出所有函数定义和关键计算块
3. **添加 docstring** — 为每个函数编写带 LaTeX 公式的中文文档（完整版）
4. **替换注释** — 用带 LaTeX 公式的新注释替换现有注释
5. **注释数学表达式** — 为关键计算添加行内 LaTeX 注释
6. **文档化循环/条件** — 解释迭代和分支的物理意义
7. **验证** — 确保 LaTeX 语法正确，反斜杠转义无误
