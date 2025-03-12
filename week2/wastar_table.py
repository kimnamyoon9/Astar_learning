import numpy as np
import time
import heapq
import random
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use("Qt5Agg")  # 또는 "TkAgg"

def map_build1(grid, random_state):
    start = grid[5][5]
    end = grid[90][90]
    start.make_start()
    end.make_end()
    total_cells = len(grid) * len(grid[0])
    num_barriers = int(total_cells * 0.3)
    random.seed(random_state)

    barrier_positions = random.sample(
        [(i, j) for i in range(len(grid)) for j in range(len(grid[0]))
         if (i, j) != (5, 5) and (i, j) != (90, 90)], num_barriers
    )

    for row, col in barrier_positions:
        grid[row][col].make_barrier()

    return start, end

class Grid:
    def __init__(self, row, col, total_rows):
        self.row = row
        self.col = col
        self.color = "WHITE"
        self.total_rows = total_rows
        self.neighbors = []
    
    def get_pos(self):
        return self.row, self.col

    def is_barrier(self):
        return self.color == "BLACK"
    
    def make_start(self):
        self.color = "ORANGE"
    
    def make_end(self):
        self.color = "TURQUOISE"
    
    def make_barrier(self):
        self.color = "BLACK"

    def make_closed(self):
        self.color = "RED"
    
    def make_open(self):
        self.color = "GREEN"

def heuristic(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return (abs(x1 - x2) + abs(y1 - y2)) * 7 + abs(abs(x1 - x2) - abs(y1 - y2)) * 3

def get_neighbors(grid, cell):
    neighbors = []
    rows = cell.total_rows
    row, col = cell.row, cell.col
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dr, dc in directions:
        r, c = row + dr, col + dc
        if 0 <= r < rows and 0 <= c < len(grid[0]):
            if abs(dr) + abs(dc) == 2:  # 대각선 이동
                if not (grid[row][c].is_barrier() or grid[r][col].is_barrier() or grid[r][c].is_barrier()):
                    neighbors.append(grid[r][c])
            else:  # 직선 이동
                if not grid[r][c].is_barrier():
                    neighbors.append(grid[r][c])
    
    return neighbors

def cost(current, neighbor):
    return 10 if current.row == neighbor.row or current.col == neighbor.col else 14

def calculate_path_length(came_from, start, end):
    path_length = 0
    current = end
    while current in came_from:
        prev = came_from[current]
        path_length += cost(prev, current)
        current = prev
    return path_length / 10  # 정규화된 거리

def a_star_algorithm(grid, start, end, weight):
    count = 0
    open_set = []
    heapq.heappush(open_set, (0, count, start))
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = weight * heuristic(start.get_pos(), end.get_pos())
    open_set_hash = {start.get_pos()}
    came_from = {}
    
    while open_set:
        current = heapq.heappop(open_set)[2]
        
        if current == end:
            return came_from
        
        for neighbor in get_neighbors(grid, current):
            temp_g_score = g_score[current] + cost(current, neighbor)
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + weight * heuristic(neighbor.get_pos(), end.get_pos())
                if neighbor.get_pos() not in open_set_hash:
                    count += 1
                    heapq.heappush(open_set, (f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor.get_pos())
                    neighbor.make_open()
        
        current.make_closed()
    
    return None

def run_experiment(grid_size, weight, random_state, num_trials):
    grid = [[Grid(i, j, grid_size) for j in range(grid_size)] for i in range(grid_size)]
    start, end = map_build1(grid, random_state)
    
    for row in grid:
        for spot in row:
            spot.neighbors = get_neighbors(grid, spot)
    
    results = np.zeros((num_trials, 4))  # 시행횟수에 따라 동적으로 행렬 크기 조정
    
    for i in range(num_trials):
        start_time = time.time()
        came_from = a_star_algorithm(grid, start, end, weight)
        elapsed_time = (time.time() - start_time) * 1000  # 소요 시간(ms) 단위 변환
        
        searched_area = sum(1 for row in grid for spot in row if spot.color in ["RED", "GREEN"])
        path_length = calculate_path_length(came_from, start, end) if came_from else 0
        
        results[i] = [i + 1, elapsed_time, searched_area, path_length]  # 시행순서, 소요시간(ms), 탐색범위, 경로 길이 저장
    
    return results
'''
def plot_time_distribution(results):
    times = results[:, 1]  # 소요 시간(ms) 데이터 가져오기
    plt.figure(figsize=(8, 6))
    plt.hist(times, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel("daration (ms)")
    plt.ylabel("steps")
    plt.title("time_distribution")
    plt.grid(True)
    #plt.show()
    plt.savefig("time_distribution.png")  # PNG 파일로 저장
    print("그래프가 time_distribution.png 파일로 저장되었습니다.")
'''
# 실행 예시
num_trials = 5000  # 시행 횟수를 조정 가능
experiment_results = run_experiment(100, 5, 20, num_trials)
np.set_printoptions(suppress=True, precision=6)  # 지수 표기법 제거 및 소수점 자리 설정
#print("실험 결과 (시행순서, 소요시간(ms), 탐색범위, 경로 길이):\n", experiment_results)


# 소요 시간 분포 시각화
#plot_time_distribution(experiment_results)
