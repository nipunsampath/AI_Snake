import math
from random import randint
import pygame


def roundup(x):
    return int(math.ceil(x / 10.0)) * 10


class Snake:
    def __init__(self, height=500, width=500, is_gui=False):
        """Initialize Components"""
        self.snake = []
        self.score = 0
        self.board = {'height': height, 'width': width}
        self.done = True
        self.is_gui = is_gui
        self.food = []
        self.done = False
        self.window = pygame.display.set_mode((self.board['width'] + 20, self.board['height'] + 20))
        self.snake_steps = 10
        self.score_height = 25
        self.is_game_over = False

    def init_snake(self):
        """Initialize the snake"""
        x = roundup(randint(5, self.board["width"] - 5))
        y = roundup(randint(5, self.board["height"] - 5))
        orientation = randint(0, 1)
        horizontal = orientation == 0
        init_length = 3

        for i in range(init_length):
            if horizontal:
                point = [x + self.snake_steps * i, y]
            else:
                point = [x, y + self.snake_steps * i]

            self.snake.insert(0, point)

    def create_food(self):
        """Generate food"""
        food = []
        while not food:
            x = roundup(randint(10, self.board['width']))
            y = roundup(randint(10, self.board['height']))
            # food = [randint(1, self.board['width']), randint(1, self.board['height'])]
            food = [x, y]
            if food in self.snake:
                food = []

        self.food = food

    def init_rendering(self):
        """Initialize the game rendering process"""
        pygame.init()
        pygame.display.set_caption("AI Snake")
        self.render()

    def render(self):
        """Rendering the frames"""
        pygame.font.init()
        self.window.fill((0, 0, 0))
        # b = '\U0001F600'
        header_font = pygame.font.SysFont('Comic Sans MS', 18)
        normal_font = pygame.font.SysFont('Courier New', 15)

        score = header_font.render('Score: ' + str(self.score) + ' ', False, (255, 255, 0))
        # score = header_font.render('Score: ' + str(self.score) + ' ' + 'Snake: ' + str(self.snake) + 'food: ' + str(
        # self.food), False, (255, 255, 255))
        self.score_height = score.get_height()
        self.window.blit(score, (0, 2))
        food = header_font.render("O", False, (255, 0, 0))
        self.window.blit(food, (self.food[0], self.food[1]))
        snake_head = normal_font.render("O", False, (255, 255, 255))
        snake_body = normal_font.render("o", False, (255, 255, 255))

        for i, point in enumerate(self.snake):
            if i == 0:
                self.window.blit(snake_head, (point[0], point[1]))
            else:
                self.window.blit(snake_body, (point[0], point[1]))
        pygame.display.update()

    def move(self, key):
        """Move the Snake"""
        # 0 - UP
        # 1 - RIGHT
        # 2 - DOWN
        # 3 - LEFT
        if not self.is_game_over:
            self.create_new_frame(key)
            if self.food_eaten():
                self.score += 1
                self.create_food()
            else:
                self.remove_last_frame()

            if self.is_gui:
                pygame.time.wait(100)
                self.render()
            self.check_collisions()
            if self.done:
                self.end_game()
            return self.generate_observations()

    def generate_observations(self):
        return self.done, self.score, self.snake, self.food

    def start(self):
        """Start the game"""
        self.init_snake()
        self.create_food()
        if self.is_gui:
            self.init_rendering()
        return self.generate_observations()

    def end_game(self):
        """End the Game"""
        if self.is_gui:
            self.destroy_rendering()
            # raise Exception("Game Over")
            self.is_game_over = True
            print("Game Over!")

    def create_new_frame(self, key):
        new_point = [self.snake[0][0], self.snake[0][1]]
        if key == 0:
            new_point[1] -= self.snake_steps
        elif key == 1:
            new_point[0] += self.snake_steps
        elif key == 2:
            new_point[1] += self.snake_steps
        elif key == 3:
            new_point[0] -= self.snake_steps
        self.snake.insert(0, new_point)

    def remove_last_frame(self):
        self.snake.pop()

    def food_eaten(self):
        return self.snake[0] == self.food

    def check_collisions(self):
        """Check for collisions"""
        if (self.snake[0][0] < 0 or
                self.snake[0][0] > self.board["width"] or
                self.snake[0][1] < self.score_height - 6 or
                self.snake[0][1] > self.board["height"] or
                self.snake[0] in self.snake[1:]):
            self.done = True
            # score_height - 6 is kind of adjustment to get precise boundary

    @staticmethod
    def destroy_rendering():
        pygame.quit()


if __name__ == "__main__":
    game = Snake(is_gui=True)
    game.start()
    run = True
    while run:
        if game.is_game_over:
            break

        pygame.time.delay(100)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        keys = pygame.key.get_pressed()

        #  To move the snake using arrow keys
        if keys[pygame.K_LEFT]:
            game.move(3)
        if keys[pygame.K_RIGHT]:
            game.move(1)

        if keys[pygame.K_UP]:
            game.move(0)

        if keys[pygame.K_DOWN]:
            game.move(2)

        # # automatic movement
        # for _ in range(20):
        #     game.move(randint(0, 3))

    pygame.quit()
