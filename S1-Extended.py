"""
S1-Extended: 위상결합 파라미터 민감도 분석
- k_coupling 스윕: [1, 5, 10, 20, 50]
- 교란 시점 스윕: [t=1.0s (s≈0.45, 초반), t=2.0s (s≈0.20, 후반)]
- 측정 지표: 재동기화 시간, 최종 |Δs|, 오른팔 위치 오차

논문 활용: 
  - GNN이 k를 자동 결정해야 하는 이유의 실험 근거
  - "k=10이 경험적 최적값" 정당화
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from itertools import product

# ─────────────────────────────────────────
#  DMP1D 클래스 (S1과 동일)
# ─────────────────────────────────────────
class DMP1D:
    def __init__(self, alpha_z=25.0, beta_z=6.25, alpha_s=4.0, tau=1.0, dt=0.001):
        self.alpha_z = alpha_z
        self.beta_z  = beta_z
        self.alpha_s = alpha_s
        self.tau     = tau
        self.dt      = dt

    def reset(self, y0, goal):
        self.s   = 1.0
        self.y   = y0
        self.dy  = 0.0
        self.ddy = 0.0
        self.goal = goal

    def step(self, coupling_s=0.0):
        ds = (-self.alpha_s * self.s + coupling_s) / self.tau
        self.s = max(0.0, self.s + ds * self.dt)
        self.ddy = (
            self.alpha_z * (self.beta_z * (self.goal - self.y)
                            - self.tau * self.dy)
        ) / (self.tau ** 2)
        self.dy += self.ddy * self.dt
        self.y  += self.dy  * self.dt
        return self.y, self.s


# ─────────────────────────────────────────
#  단일 실험 실행 함수
# ─────────────────────────────────────────
def run_experiment(k, t_perturb, delta_s=0.15, T=5.0, tau=5.0, dt=0.001):
    """
    반환: (t_arr, s_L, s_R, delta_s_arr, y_R, resync_time, success)
    """
    dmp_L = DMP1D(tau=tau, dt=dt)
    dmp_R = DMP1D(tau=tau, dt=dt)
    dmp_L.reset(0.0, 0.2)
    dmp_R.reset(0.0, 0.2)

    n_steps = int(T / dt)
    t_arr   = np.arange(n_steps) * dt

    s_L_arr, s_R_arr = np.zeros(n_steps), np.zeros(n_steps)
    y_R_arr          = np.zeros(n_steps)
    ds_arr           = np.zeros(n_steps)

    perturbed = False

    for i in range(n_steps):
        t = t_arr[i]
        if t >= t_perturb and not perturbed:
            dmp_R.s = max(0.0, dmp_R.s - delta_s)
            perturbed = True

        d = dmp_L.s - dmp_R.s
        y_L, s_L = dmp_L.step(coupling_s=-k * d)
        y_R, s_R = dmp_R.step(coupling_s=+k * d)

        s_L_arr[i] = s_L
        s_R_arr[i] = s_R
        y_R_arr[i] = y_R
        ds_arr[i]  = d

    # 재동기화 시간 계산
    post_idx  = int(t_perturb / dt)
    resync_time = None
    for i, val in enumerate(np.abs(ds_arr[post_idx:])):
        if val < 0.05:
            resync_time = i * dt
            break

    success = resync_time is not None and resync_time <= 0.5
    final_ds = abs(ds_arr[-1])

    return t_arr, s_L_arr, s_R_arr, ds_arr, y_R_arr, resync_time, success, final_ds


# ─────────────────────────────────────────
#  실험 파라미터 그리드
# ─────────────────────────────────────────
k_list      = [1, 5, 10, 20, 50]
t_perturb_list = [1.0, 2.0]
delta_s     = 0.15
tau         = 5.0

# 이론값 계산: λ = (α_s + 2k) / τ
alpha_s = 4.0
print("─" * 60)
print(f"{'k':>5} | {'λ=이론 감쇠율':>14} | {'이론 재동기화 시간(s)':>20}")
print("─" * 60)
for k in k_list:
    lam  = (alpha_s + 2 * k) / tau
    t_th = np.log(delta_s / (0.05)) / lam  # ln(Δs0 / threshold) / λ
    print(f"{k:>5} | {lam:>14.3f} | {t_th:>20.4f}s")
print("─" * 60)


# ─────────────────────────────────────────
#  모든 조합 실행 및 결과 수집
# ─────────────────────────────────────────
results = {}
for k, t_p in product(k_list, t_perturb_list):
    key = (k, t_p)
    t_arr, s_L, s_R, ds_arr, y_R, resync, success, final_ds = run_experiment(k, t_p, delta_s, tau=tau)
    results[key] = {
        't': t_arr, 's_L': s_L, 's_R': s_R, 'ds': ds_arr, 'y_R': y_R,
        'resync': resync, 'success': success, 'final_ds': final_ds
    }
    status = "PASS ✅" if success else "FAIL ❌"
    resync_str = f"{resync:.3f}s" if resync else "미수렴"
    print(f"k={k:>2}, t_perturb={t_p}s → 재동기화: {resync_str:>8}  |  최종 |Δs|: {final_ds:.6f}  |  {status}")

print()


# ─────────────────────────────────────────
#  그래프 1: k별 위상 차이 비교 (2행 × 5열)
# ─────────────────────────────────────────
fig1, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=False)
fig1.suptitle('S1 민감도: k_coupling × 교란 시점 — 위상 차이 |Δs| 수렴', fontsize=14, fontweight='bold')

for col, k in enumerate(k_list):
    for row, t_p in enumerate(t_perturb_list):
        ax = axes[row, col]
        r  = results[(k, t_p)]

        ax.plot(r['t'], np.abs(r['ds']), color='purple', linewidth=1.5)
        ax.axhline(0.05, color='green', linestyle='--', linewidth=1.2)
        ax.axvline(t_p,  color='orange', linestyle=':', linewidth=1.5)

        if r['resync']:
            ax.axvline(t_p + r['resync'], color='green', linestyle='-', linewidth=1.5)
            ax.set_title(f"k={k}, t_p={t_p}s\n재동기화: {r['resync']:.3f}s ✅",
                         fontsize=9, color='green')
        else:
            ax.set_title(f"k={k}, t_p={t_p}s\n미수렴 ❌", fontsize=9, color='red')

        ax.set_xlim(0, 5)
        ax.set_ylim(-0.01, 0.22)
        ax.set_xlabel('시간 (s)', fontsize=8)
        if col == 0:
            ax.set_ylabel(f't_p={t_p}s\n|Δs|', fontsize=9)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('s1_sensitivity_phase.png', dpi=150, bbox_inches='tight')
print("저장: s1_sensitivity_phase.png")


# ─────────────────────────────────────────
#  그래프 2: 결과 요약 히트맵
# ─────────────────────────────────────────
fig2, (ax_heat1, ax_heat2) = plt.subplots(1, 2, figsize=(12, 4))
fig2.suptitle('S1 민감도 요약 히트맵', fontsize=13, fontweight='bold')

resync_matrix = np.zeros((2, 5))
success_matrix = np.zeros((2, 5))

for ci, k in enumerate(k_list):
    for ri, t_p in enumerate(t_perturb_list):
        r = results[(k, t_p)]
        resync_matrix[ri, ci]  = r['resync'] if r['resync'] else 999.0
        success_matrix[ri, ci] = 1.0 if r['success'] else 0.0

# 재동기화 시간 히트맵
im1 = ax_heat1.imshow(resync_matrix, cmap='RdYlGn_r', aspect='auto',
                       vmin=0, vmax=1.0)
ax_heat1.set_xticks(range(5)); ax_heat1.set_xticklabels(k_list)
ax_heat1.set_yticks(range(2)); ax_heat1.set_yticklabels([f't_p={t}s' for t in t_perturb_list])
ax_heat1.set_xlabel('k_coupling'); ax_heat1.set_title('재동기화 시간 (s)\n(붉을수록 느림 / 999=미수렴)')
for ri in range(2):
    for ci in range(5):
        val = resync_matrix[ri, ci]
        txt = f"{val:.3f}" if val < 999 else "미수렴"
        ax_heat1.text(ci, ri, txt, ha='center', va='center',
                      fontsize=10, color='black', fontweight='bold')
plt.colorbar(im1, ax=ax_heat1)

# 성공 여부 히트맵
im2 = ax_heat2.imshow(success_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax_heat2.set_xticks(range(5)); ax_heat2.set_xticklabels(k_list)
ax_heat2.set_yticks(range(2)); ax_heat2.set_yticklabels([f't_p={t}s' for t in t_perturb_list])
ax_heat2.set_xlabel('k_coupling'); ax_heat2.set_title('성공 여부\n(녹색=PASS, 적색=FAIL)')
for ri in range(2):
    for ci in range(5):
        txt = "PASS" if success_matrix[ri, ci] == 1 else "FAIL"
        ax_heat2.text(ci, ri, txt, ha='center', va='center',
                      fontsize=11, color='black', fontweight='bold')
plt.colorbar(im2, ax=ax_heat2)

plt.tight_layout()
plt.savefig('s1_sensitivity_heatmap.png', dpi=150, bbox_inches='tight')
print("저장: s1_sensitivity_heatmap.png")


# ─────────────────────────────────────────
#  그래프 3: 이론값 vs 실험값 검증 (t_p=2.0s 기준)
# ─────────────────────────────────────────
fig3, ax = plt.subplots(figsize=(8, 5))
theory_times = []
exp_times_t2  = []
exp_times_t1  = []

for k in k_list:
    lam  = (alpha_s + 2 * k) / tau
    t_th = np.log(delta_s / 0.05) / lam
    theory_times.append(t_th)
    r2 = results[(k, 2.0)]
    r1 = results[(k, 1.0)]
    exp_times_t2.append(r2['resync'] if r2['resync'] else np.nan)
    exp_times_t1.append(r1['resync'] if r1['resync'] else np.nan)

ax.plot(k_list, theory_times, 'k--o', linewidth=2, markersize=8, label='이론값 (해석해)')
ax.plot(k_list, exp_times_t2, 'r-s',  linewidth=2, markersize=8, label='실험값 t_p=2.0s')
ax.plot(k_list, exp_times_t1, 'b-^',  linewidth=2, markersize=8, label='실험값 t_p=1.0s')
ax.axhline(0.5, color='green', linestyle=':', linewidth=2, label='성공 기준 0.5s')
ax.axvline(10,  color='orange', linestyle=':', linewidth=1.5, label='기본값 k=10')

ax.set_xlabel('k_coupling', fontsize=12)
ax.set_ylabel('재동기화 시간 (s)', fontsize=12)
ax.set_title('이론값 vs 실험값 — 재동기화 시간 비교\n(이론과 실험이 일치하면 수식 구현이 올바른 것)', fontsize=11)
ax.set_xscale('log')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('s1_theory_vs_experiment.png', dpi=150, bbox_inches='tight')
print("저장: s1_theory_vs_experiment.png")

print("\n─── S1 민감도 분석 완료 ───")
