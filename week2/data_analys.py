import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wastar_table import run_experiment

# 실행할 실험 설정
num_trials = 30  # 시행 횟수

def main():
    print("실험 실행 중...")
    all_results = []
    
    # 랜덤 시드를 1부터 1000까지 변경하면서 실행
    for random_seed in range(0, 1000):
        print(f"실험 진행 중... 랜덤 시드: {random_seed}")
        results = run_experiment(100, 5, random_seed, num_trials)
        
        # 실행 결과에 랜덤 시드 정보 추가
        for row in results:
            all_results.append([random_seed] + list(row))
    
    # 결과를 DataFrame으로 변환
    df = pd.DataFrame(all_results, columns=["Random Seed", "Trial", "Time (ms)", "Searched Area", "Path Length"])
    '''
    # 탐색 범위가 6000이 넘는 랜덤 시드 찾기
    high_searched_area = df[df["Searched Area"] > 6000]["Random Seed"].unique()
    if len(high_searched_area) > 0:
        print("탐색 범위가 6000을 초과하는 랜덤 시드:", high_searched_area)
    else:
        print("탐색 범위가 6000을 초과하는 랜덤 시드가 없습니다.")
    '''

    # CSV 파일로 저장
    df.to_csv("experiment_results.csv", index=False)
    print("실험 결과가 experiment_results.csv 파일로 저장되었습니다.")
    
    # 데이터 분석 및 시각화
    plot_results(df)

def plot_results(df):
    plt.figure(figsize=(8, 6))
    plt.hist(df["Time (ms)"], bins=200, edgecolor='black', alpha=0.7)
    # x축 범위 제한 (집중된 영역인 1.5 ~ 3.5ms로 조정)
    plt.xlim(2, 3)
    # x축 눈금 간격 조정 (0.1 간격으로 더 보기 쉽게)
    plt.xticks(np.arange(2, 3.1, 0.1))
    plt.xlabel("Duration (ms)")
    plt.ylabel("Count")
    plt.title("Time Distribution")
    plt.grid(True)
    plt.savefig("time_distribution.png")
    print("그래프가 time_distribution.png 파일로 저장되었습니다.")
    
    plt.figure(figsize=(8, 6))
    plt.hist(df["Searched Area"], bins=50, edgecolor='black', alpha=0.7)
    # x축 범위 제한 (예: 최소 0, 최대 200)
    plt.xlim(0,1300)
    # x축 눈금 간격 조정 (예: 10 간격으로 표시)
    plt.xticks(np.arange(0,1300, 100))
    plt.xlabel("Searched Area")
    plt.ylabel("Count")
    plt.title("Searched Area Distribution")
    plt.grid(True)
    plt.savefig("searched_area_distribution.png")
    print("그래프가 searched_area_distribution.png 파일로 저장되었습니다.")
    
    plt.figure(figsize=(8, 6))
    plt.hist(df["Path Length"], bins=30, edgecolor='black', alpha=0.7)  
    # x축 범위 제한 (예: 최소 0, 최대 200)
    plt.xlim(130, 230)
    # x축 눈금 간격 조정 (예: 10 간격으로 표시)
    plt.xticks(np.arange(130, 230, 10))
    plt.xlabel("Path Length")
    plt.ylabel("Count")
    plt.title("Path Length Distribution")
    plt.grid(True)
    plt.savefig("path_length_distribution.png")
    print("그래프가 path_length_distribution.png 파일로 저장되었습니다.")

if __name__ == "__main__":
    main()
