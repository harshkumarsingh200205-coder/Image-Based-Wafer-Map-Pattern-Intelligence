# Spatial Feature Engineering: Mathematical Formulations

Treating the wafer map as a continuous 2D spatial coordinate system $\mathcal{W} \subset \mathbb{R}^2$ with disc center $\mathbf{c} = (x_0, y_0)$ and radius $R$, each defective die is represented at pixel location $\mathbf{p}_k = (x_k, y_k)$.

---

## 1. Radial Annular Density ($D_{\text{rad}}$)

The wafer disc is partitioned into $K$ concentric rings of equal radial thickness:
$$\text{Ring}_i = \left\{ \mathbf{p} \in \mathcal{W} \;\Big|\; \frac{i-1}{K} R \le \|\mathbf{p} - \mathbf{c}\|_2 < \frac{i}{K} R \right\}, \quad i = 1, \dots, K$$

The radial defect density for ring $i$ is defined as:
$$D_{\text{rad}}(i) = \frac{\sum_{\mathbf{p} \in \text{Ring}_i} M(\mathbf{p})}{|\text{Ring}_i|}$$
where $M(\mathbf{p}) \in \{0, 1\}$ is the binary defect mask and $|\text{Ring}_i|$ is the total die count in ring $i$.

---

## 2. Angular Sector Distribution ($D_{\text{ang}}$)

The wafer is partitioned into $M$ angular pie slices (clock sectors):
$$\text{Sector}_j = \left\{ \mathbf{p} \in \mathcal{W} \;\Big|\; \frac{2\pi(j-1)}{M} \le \text{atan2}(y - y_0, x - x_0) < \frac{2\pi j}{M} \right\}, \quad j = 1, \dots, M$$

$$D_{\text{ang}}(j) = \frac{\sum_{\mathbf{p} \in \text{Sector}_j} M(\mathbf{p})}{|\text{Sector}_j|}$$

---

## 3. Center-to-Edge Ratio ($R_{\text{c/e}}$)

Let $\mathcal{W}_{\text{center}} = \{\mathbf{p} \mid \|\mathbf{p} - \mathbf{c}\|_2 \le 0.3R\}$ and $\mathcal{W}_{\text{edge}} = \{\mathbf{p} \mid 0.7R \le \|\mathbf{p} - \mathbf{c}\|_2 \le R\}$.

$$R_{\text{c/e}} = \frac{D(\mathcal{W}_{\text{center}})}{D(\mathcal{W}_{\text{edge}}) + \epsilon}$$
where $\epsilon = 10^{-6}$ is a numerical stabilizer.
- If $R_{\text{c/e}} \gg 1 \implies$ strong Center failure mode.
- If $R_{\text{c/e}} \ll 1 \implies$ strong Edge failure mode.

---

## 4. Inertia Tensor & Geometric Eccentricity

For the set of defect coordinates $\{\mathbf{p}_k\}_{k=1}^N$, we compute the second spatial central moments:
$$\mu_{xx} = \frac{1}{N}\sum (x_k - \bar{x})^2, \quad \mu_{yy} = \frac{1}{N}\sum (y_k - \bar{y})^2, \quad \mu_{xy} = \frac{1}{N}\sum (x_k - \bar{x})(y_k - \bar{y})$$

The eigenvalues $\lambda_1 \ge \lambda_2$ of the covariance matrix $\begin{bmatrix} \mu_{xx} & \mu_{xy} \\ \mu_{xy} & \mu_{yy} \end{bmatrix}$ characterize the defect distribution:
- **Eccentricity**: $e = \sqrt{1 - \frac{\lambda_2}{\lambda_1}}$ (Approaches $1.0$ for linear **Scratch** defects).
- **Inertia Ratio**: $\frac{\lambda_1}{\lambda_2}$.
