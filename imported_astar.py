import pygame
import numpy as np
import math
import time
from queue import PriorityQueue

WIDTH = 900
WIN = pygame.display.set_mode((WIDTH, 900))

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)
BROWN = (165,42,42)

def map_build(grid):
    grid[9][9].make_start()
    grid[81][27].make_end()
    for col in range(0, 63):
        grid[45][col].make_barrier()

class grids:

    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_grass(self):
        return self.color == BROWN

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ORANGE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_grass(self):
        self.color = BROWN

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = PURPLE

    def draws(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): # DOWN
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # UP
            self.neighbors.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # RIGHT
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])
        
        if (self.row < self.total_rows - 1 and self.col < self.total_rows - 1 and
            not grid[self.row + 1][self.col + 1].is_barrier() and
            not grid[self.row + 1][self.col].is_barrier() and
            not grid[self.row][self.col + 1].is_barrier()
            ): # DOWN RIGHT
            self.neighbors.append(grid[self.row + 1][self.col + 1])

        if (self.row < self.total_rows - 1 and self.col > 0 and
            not grid[self.row + 1][self.col - 1].is_barrier() and
            not grid[self.row + 1][self.col].is_barrier() and
            not grid[self.row][self.col - 1].is_barrier()
            ): # DOWN LEFT
            self.neighbors.append(grid[self.row + 1][self.col - 1])

        if (self.row > 0 and self.col < self.total_rows - 1 and
            not grid[self.row - 1][self.col + 1].is_barrier() and
            not grid[self.row - 1][self.col].is_barrier() and
            not grid[self.row][self.col + 1].is_barrier()
            ): # UP RIGHT
            self.neighbors.append(grid[self.row - 1][self.col + 1])

        if (self.row > 0 and self.col > 0 and
            not grid[self.row - 1][self.col - 1].is_barrier() and
            not grid[self.row - 1][self.col].is_barrier() and
            not grid[self.row][self.col - 1].is_barrier()
            ): # UP LEFT
            self.neighbors.append(grid[self.row - 1][self.col - 1])

    def make_grid(rows, width):
        grid = []
        gap = width // rows
        for i in range(rows):
            grid.append([])
            for j in range(rows):
                spot = grids(i, j, gap, rows)
                grid[i].append(spot)
        return grid
    
    def h(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return abs(x1 - x2) + abs(y1 - y2)
    
    def get_clicked_pos(pos, rows, width):
        gap = width // rows
        y, x = pos

        row = y // gap
        col = x // gap

        return row, col
    
    def draw_grid(win, rows, width):
        gap = width // rows
        for i in range(rows):
            pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
            for j in range(rows):
                pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

    def draw(win, grid, rows, width):
        win.fill(WHITE)
        for row in grid:
            for spot in row:
                spot.draws(win)

        grids.draw_grid(win, rows, width)
        pygame.display.update()

    def __lt__(self, other):
	    return False
    
class a_star:
    
    def get_neighbors(grid,cell):
        neighbors=[]
        if cell.row < cell.total_rows - 1 and not grid[cell.row + 1][cell.col].is_barrier(): # DOWN
            neighbors.append(grid[cell.row + 1][cell.col])

        if cell.row > 0 and not grid[cell.row - 1][cell.col].is_barrier(): # UP
            neighbors.append(grid[cell.row - 1][cell.col])

        if cell.col < cell.total_rows - 1 and not grid[cell.row][cell.col + 1].is_barrier(): # RIGHT
            neighbors.append(grid[cell.row][cell.col + 1])

        if cell.col > 0 and not grid[cell.row][cell.col - 1].is_barrier(): # LEFT
            neighbors.append(grid[cell.row][cell.col - 1])

        if (cell.row < cell.total_rows - 1 and cell.col < cell.total_rows - 1 and
            not grid[cell.row + 1][cell.col + 1].is_barrier() and
            not grid[cell.row + 1][cell.col].is_barrier() and
            not grid[cell.row][cell.col + 1].is_barrier()
            ): # DOWN RIGHT
            neighbors.append(grid[cell.row + 1][cell.col + 1])

        if (cell.row < cell.total_rows - 1 and cell.col > 0 and
            not grid[cell.row + 1][cell.col - 1].is_barrier() and
            not grid[cell.row + 1][cell.col].is_barrier() and
            not grid[cell.row][cell.col - 1].is_barrier()
            ): # DOWN LEFT
            neighbors.append(grid[cell.row + 1][cell.col - 1])

        if (cell.row > 0 and cell.col < cell.total_rows - 1 and
            not grid[cell.row - 1][cell.col + 1].is_barrier() and
            not grid[cell.row - 1][cell.col].is_barrier() and
            not grid[cell.row][cell.col + 1].is_barrier()
            ): # UP RIGHT
            neighbors.append(grid[cell.row - 1][cell.col + 1])

        if (cell.row > 0 and cell.col > 0 and
            not grid[cell.row - 1][cell.col - 1].is_barrier() and
            not grid[cell.row - 1][cell.col].is_barrier() and
            not grid[cell.row][cell.col - 1].is_barrier()
            ): # UP LEFT
            neighbors.append(grid[cell.row - 1][cell.col - 1])

        return neighbors

    def weight(grid,grid_array,current, neighbor):
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == current and (neighbor==grid[i+1][j] or neighbor==grid[i][j+1] or neighbor==grid[i-1][j] or neighbor==grid[i][j-1]):
                    return grid_array[i][j]
                else:
                    return math.sqrt(2)

    def reconstruct_path(came_from, current, draw):
        while current in came_from:
            current = came_from[current]
            current.make_path()
            draw()

    def algorithm(draw, grid, start, end, grid_array):
        count = 0
        open_set = PriorityQueue()
        open_set.put((0, count, start))
        came_from = {}
        g_score = {spot: float("inf") for row in grid for spot in row}
        g_score[start] = 0
        f_score = {spot: float("inf") for row in grid for spot in row}
        f_score[start] = grids.h(start.get_pos(), end.get_pos())
        open_set_hash = {start}

        while not open_set.empty():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            current = open_set.get()[2]
            open_set_hash.remove(current)
            if current == end:
                a_star.reconstruct_path(came_from, end, draw)
                end.make_end()
                return True
            neighbors = a_star.get_neighbors(grid,current)
            for neighbor in neighbors:
                temp_g_score = g_score[current] + a_star.weight(grid,grid_array,current, neighbor)
                if temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + grids.h(neighbor.get_pos(), end.get_pos())
                    if neighbor not in open_set_hash:
                        count += 1
                        open_set.put((f_score[neighbor], count, neighbor))
                        open_set_hash.add(neighbor)
                        neighbor.make_open()

            draw()

            if current != start:
                current.make_closed()
        return False


def main(win, width):
    ROWS = 90
    grid = grids.make_grid(ROWS, width)
    grid_array= np.ones((ROWS,ROWS))
    start = None
    end = None
    count=0
    run = True
    map_build(grid)

    while run:
        grids.draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            flag=False
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]: # LEFT
                pos = pygame.mouse.get_pos()
                row, col = grids.get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                if not start and spot != end:
                    grid_array[row][col]=1
                    start = spot
                    start.make_start()

                elif not end and spot != start:
                    grid_array[row][col]=1
                    end = spot
                    end.make_end()

                elif spot != end and spot != start and not grid[row][col].is_grass():
                    grid_array[row][col]+=1
                    spot.make_grass()

                elif grid[row][col].is_grass():
                    grid_array[row][col]+=2
                    spot.make_barrier()
            elif pygame.mouse.get_pressed()[2]: # RIGHT
                    pos = pygame.mouse.get_pos()
                    row, col = grids.get_clicked_pos(pos, ROWS, width)
                    spot = grid[row][col]
                    spot.reset()
                    if spot == start:
                        start = None
                    elif spot == end:
                        end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    start_time = time.time()
                    for row in grid:
                        for spot in row:
                                spot.update_neighbors(grid)

                    a_star.algorithm(lambda: grids.draw(win, grid, ROWS, width), grid, start, end, grid_array)
                    count+=1
                    end_time = time.time()
                    print(end_time - start_time)
                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = grids.make_grid(ROWS, width)
                

    pygame.quit()


main(WIN, WIDTH)
