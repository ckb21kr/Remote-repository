# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.fft import rfft, rfftfreq
import streamlit as st

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None

# ==========================================
# [0. 스트림릿 페이지 웹 레이아웃 설정]
# ==========================================
st.set_page_config(
    page_title="회전설비 CBM 상태진단 대시보드",
    page_icon="⚙️",
    layout="wide"
)

# 마이너스 기호 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False

# ==========================================
# [1. 데이터 분석 핵심 연산 엔진]
# ==========================================
def calculate_features(signal):
    """시간 영역 통계 특징값 계산"""
    signal = np.asarray(signal).ravel()
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    kurtosis = stats.kurtosis(signal, fisher=False)
    skewness = stats.skew(signal)
    crest_factor = peak / rms if rms > 0 else np.nan
    std = np.std(signal)
    mean_abs = np.mean(np.abs(signal))
    
    return {
        "mean": np.mean(signal),
        "std": std,
        "rms": rms,
        "peak": peak,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "crest_factor": crest_factor,
        "mean_abs": mean_abs,
    }

def compute_fft(signal, fs):
    """보정된 단방향 FFT 계산"""
    signal = np.asarray(signal).ravel()
    signal = signal - np.mean(signal)
    n = len(signal)
    window = np.hanning(n)
    spectrum = 2.0 * np.abs(rfft(signal * window)) / np.sum(window)
    freq = rfftfreq(n, 1 / fs)
    return freq, spectrum

def window_features(signal, fs, window_sec=0.2, step_sec=0.1):
    """시계열 데이터를 구간별로 쪼개어 특징값 추세 도출"""
    signal = np.asarray(signal).ravel()
    window = int(fs * window_sec)
    step = int(fs * step_sec)
    rows = []
    for start in range(0, len(signal) - window + 1, step):
        seg = signal[start:start + window]
        rows.append({
            "time_sec": start / fs,
            **calculate_features(seg),
        })
    return pd.DataFrame(rows)

def diagnose_faults(fault_win_df, normal_win_df, kurt_th, crest_th):
    """베이스라인 기반 고속 규칙 매핑 상태 진단"""
    normal_baseline = normal_win_df[["rms", "kurtosis", "crest_factor"]].agg(["mean", "std"])
    rms_threshold = normal_baseline.loc["mean", "rms"] + 3 * normal_baseline.loc["std", "rms"]
    
    diagnosis = fault_win_df.copy()
    cond_rms = diagnosis["rms"] > rms_threshold
    cond_kurt = diagnosis["kurtosis"] > kurt_th
    cond_crest = diagnosis["crest_factor"] > crest_th
    
    fault_count = cond_rms.astype(int) + cond_kurt.astype(int) + cond_crest.astype(int)
    
    conditions = [fault_count >= 2, fault_count == 1]
    choices = ["위험", "주의"]
    diagnosis["diagnosis"] = np.select(conditions, choices, default="정상")
    
    reason_rms = np.where(cond_rms, "RMS 증가, ", "")
    reason_kurt = np.where(cond_kurt, "충격성 증가, ", "")
    reason_crest = np.where(cond_crest, "Crest Factor 증가, ", "")
    
    reasons_combined = reason_rms + reason_kurt + reason_crest
    diagnosis["reason"] = pd.Series(reasons_combined).str.rstrip(", ").replace("", "-")
    
    return diagnosis, rms_threshold

# ==========================================
# [2. 웹 서버 구동용 가상 데모 데이터 생성기]
# ==========================================
@st.cache_data
def generate_synthetic_data(fs, duration=2.0):
    """서버에 기본 파일이 없어도 즉시 시연 가능하도록 만드는 가상 신호 발생기"""
    t = np.arange(0, duration, 1/fs)
    normal = np.random.normal(0, 0.15, len(t))
    # 이상 신호: 강한 주기적 충격성 신호(10Hz)와 노이즈 합성
    fault = np.random.normal(0, 0.15, len(t)) + 1.2 * (np.sin(2 * np.pi * 10 * t) > 0.97).astype(float) * np.random.normal(1.0, 0.3, len(t))
    return normal, fault

# ==========================================
# [3. 대시보드 UI 및 사이드바 컨트롤러]
# ==========================================
st.title("⚙️ 예방정비를 위한 진동 기반 설비 예후 진단 대시보드")
st.markdown("본 시스템은 정상 상태 데이터의 베이스라인을 기준으로 설비의 **시간 대역 변동**, **주파수 성분(FFT)** 및 **CBM 지침**을 실시간 분석합니다.")

# 사이드바 설정 영역
st.sidebar.header("🛠️ 신호 처리 및 진단 설정")
fs = st.sidebar.number_input("샘플링 주파수 (FS, Hz)", value=12000, step=1000)
max_freq = st.sidebar.slider("FFT 시각화 최대 범위 (Hz)", 100, fs//2, 1000)

st.sidebar.subheader("🚨 규칙 기반 임계값(Threshold) 조정")
kurtosis_threshold = st.sidebar.slider("Kurtosis(첨도) 주의 임계치", 2.0, 10.0, 5.0, 0.5)
crest_threshold = st.sidebar.slider("Crest Factor(파고율) 주의 임계치", 2.0, 10.0, 4.0, 0.5)

# 데이터 불러오기 방식 정의
data_mode = st.radio("데이터 입력 모드 선택", ["📢 테스트용 시뮬레이션 데모 데이터 가동", "📂 분석 대상 MAT 파일 직접 업로드"])

normal_signal, fault_signal = None, None

if data_mode == "📢 테스트용 시뮬레이션 데모 데이터 가동":
    normal_signal, fault_signal = generate_synthetic_data(fs)
    st.success("🎯 시뮬레이션 진동 데이터셋 로드 완료!")
else:
    st.info("CWRU 형태의 정상/이상 진동 가속도 데이터(.mat)를 업로드하세요.")
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        normal_file = st.file_uploader("정상 상태 MAT 파일", type=["mat"])
    with col_up2:
        fault_file = st.file_uploader("이상 상태 MAT 파일", type=["mat"])
        
    if normal_file and fault_file and loadmat:
        try:
            mat_n = loadmat(normal_file)
            mat_f = loadmat(fault_file)
            
            keys_n = [k for k in mat_n.keys() if not k.startswith("__")]
            keys_f = [k for k in mat_f.keys() if not k.startswith("__")]
            
            col_k1, col_k2 = st.columns(2)
            with col_k1:
                sel_k_n = st.selectbox("정상 진동 시계열 변수(Key) 매핑", keys_n)
            with col_k2:
                sel_k_f = st.selectbox("이상 진동 시계열 변수(Key) 매핑", keys_f)
                
            normal_signal = mat_n[sel_k_n].ravel()
            fault_signal = mat_f[sel_k_f].ravel()
            st.success("🚀 파일 커스텀 데이터 파싱 성공!")
        except Exception as e:
            st.error(f"MAT 파일 해석 실패: {e}. 데이터 구조를 확인하세요.")

# ==========================================
# [4. 분석 파이프라인 구동 및 웹 시각화]
# ==========================================
if normal_signal is not None and fault_signal is not None:
    
    # 윈도우 특징 및 진단 연산 실행
    normal_win = window_features(normal_signal, fs)
    fault_win = window_features(fault_signal, fs)
    normal_win["state"] = "normal"
    fault_win["state"] = "fault"
    
    diagnosis_df, rms_threshold = diagnose_faults(
        fault_win, normal_win, kurt_th=kurtosis_threshold, crest_th=crest_threshold
    )
    
    # 인터페이스 레이아웃 분할용 탭 구조 생성
    tab1, tab2, tab3 = st.tabs(["📊 파형 & 피처 공간 비교", "⚡ 주파수 도메인 (FFT)", "🚨 추세 모니터링 & CBM 레포트"])
    
    # ---- 탭 1: 시간 영역 분석 ----
    with tab1:
        st.subheader("⏱️ 원시 진동 가속도 파형 비교 (초기 0.2초)")
        n_plot = min(len(normal_signal), int(fs * 0.2))
        time_axis = np.arange(n_plot) / fs
        
        fig_time, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        ax1.plot(time_axis, normal_signal[:n_plot], color="dodgerblue")
        ax1.set_title("정상 상태 진동 파형 (안정적인 운전 진폭)")
        ax1.set_ylabel("G's")
        ax1.grid(True, alpha=0.3, linestyle="--")
        
        ax2.plot(time_axis, fault_signal[:n_plot], color="crimson")
        ax2.set_title("이상 상태 진동 파형 (주기적 충격 격돌 피크 발생)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("G's")
        ax2.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()
        st.pyplot(fig_time)
        
        st.subheader("📋 전역 통계 특징값 데이터 테이블")
        feat_n = calculate_features(normal_signal)
        feat_f = calculate_features(fault_signal)
        global_features = pd.DataFrame([
            {"설비 상태": "정상 (Normal Baseline)", **feat_n},
            {"설비 상태": "이상 (Detected Fault)", **feat_f}
        ]).set_index("설비 상태")
        st.dataframe(global_features.style.format(precision=4), use_container_width=True)

    # ---- 탭 2: 주파수 영역 분석 ----
    with tab2:
        st.subheader("🔮 진폭 복원 주파수 스펙트럼 분석 (FFT)")
        st.caption("물리적인 에너지 손실 보정과 해닝 윈도우 필터가 결합되어 정확한 결함 주파수 에너지를 계측합니다.")
        
        freq_n, spec_n = compute_fft(normal_signal, fs)
        freq_f, spec_f = compute_fft(fault_signal, fs)
        
        mask_n = freq_n <= max_freq
        mask_f = freq_f <= max_freq
        
        fig_fft, (ax_f1, ax_f2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        ax_f1.plot(freq_n[mask_n], spec_n[mask_n], color="dodgerblue")
        ax_f1.set_title("정상 도메인 주파수 분포")
        ax_f1.set_ylabel("Amplitude")
        ax_f1.grid(True, alpha=0.3, linestyle="--")
        
        ax_f2.plot(freq_f[mask_f], spec_f[mask_f], color="crimson")
        ax_f2.set_title("이상 도메인 주파수 분포 (고주파 대역 및 결함 고유 성분 돌출)")
        ax_f2.set_xlabel("Frequency (Hz)")
        ax_f2.set_ylabel("Amplitude")
        ax_f2.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()
        st.pyplot(fig_fft)

    # ---- 탭 3: 실시간 추세 및 정비 가이드라인 ----
    with tab3:
        st.subheader("📈 실시간 윈도우 단위 정밀 상태 추세 판정")
        
        col_st1, col_st2 = st.columns([1, 2])
        with col_st1:
            st.metric("총 연산 윈도우 구간", f"{len(diagnosis_df)} 개")
            st.markdown("**종합 판정 통계 현황**")
            st.dataframe(diagnosis_df["diagnosis"].value_counts(), use_container_width=True)
        with col_st2:
            st.markdown("**설비 알람 로그 스냅샷**")
            st.dataframe(
                diagnosis_df[["time_sec", "rms", "kurtosis", "crest_factor", "diagnosis", "reason"]].head(10),
                use_container_width=True
            )
            
        # 추세 지표 차트 시각화
        st.markdown("#### 각 지표별 통계치 추이 및 판정 결과 매핑")
        for target_col in ["rms", "kurtosis", "crest_factor"]:
            fig_tr, ax_tr = plt.subplots(figsize=(12, 3))
            
            # 진단 등급별 분할 마킹
            color_dict = {"정상": "teal", "주의": "orange", "위험": "crimson"}
            for d_state, g_df in diagnosis_df.groupby("diagnosis"):
                ax_tr.plot(g_df["time_sec"], g_df[target_col], '.', color=color_dict.get(d_state, "blue"), label=f"상태: {d_state}")
                
            # 가이드 가로선 표기
            if target_col == "rms":
                ax_tr.axhline(rms_threshold, color="red", linestyle="--", label=f"RMS 3σ 임계치 ({rms_threshold:.4f})")
            elif target_col == "kurtosis":
                ax_tr.axhline(kurtosis_threshold, color="red", linestyle="--", label=f"첨도 임계치 ({kurtosis_threshold:.1f})")
            elif target_col == "crest_factor":
                ax_tr.axhline(crest_threshold, color="red", linestyle="--", label=f"파고율 임계치 ({crest_threshold:.1f})")
                
            ax_tr.set_title(f"구간별 {target_col.upper()} 시계열 가동 트렌드")
            ax_tr.set_xlabel("Time (s)")
            ax_tr.set_ylabel(target_col)
            ax_tr.legend(loc="upper right")
            ax_tr.grid(True, alpha=0.3, linestyle="--")
            st.pyplot(fig_tr)
            
        # CBM 리포트 다운스트림 스페이스
        st.markdown("---")
        st.subheader("📋 CBM(상태기반정비) 예방 정비 의사결정 지침")
        
        report_col1, report_col2 = st.columns(2)
        with report_col1:
            st.info(f"""
            **🟡 주의 (Caution) 조치 기준**
            * **조건**: RMS가 3σ 임계치(`{rms_threshold:.4f}`)를 돌파했거나 첨도, 파고율 중 단 한 개 지표라도 비정상 수치를 나타낼 때.
            * **현장 대응 가이드**: 장비 불시 정지의 위험은 낮으나 초기 마모나 윤활 부족 상태일 수 있습니다. 센서 데이터 계측 주기를 단축하고 다음 예방정비(PM) 셧다운 일정에 하우징 조임 상태 및 그리스 주입을 배정하세요.
            """)
        with report_col2:
            st.error(f"""
            **🔴 위험 (Alarm) 조치 기준**
            * **조건**: 에너지 지표인 RMS의 상승과 동시에 충격성 지표(첨도/파고율)가 다중 만족하며 대시보드상 '위험' 구간이 연속 검출될 때.
            * **현장 대응 가이드**: 베어링 내외륜 크랙이나 전동체 파손이 심화된 단계로, 돌발 정지(Breakdown) 확률이 매우 높습니다. 운영 파트와 조율하여 즉시 설비 가동 정비 스케줄을 확보하고 신품 교체 정비를 수행하십시오.
            """)
