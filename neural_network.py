from random import randint
import numpy as np
import tflearn
import math
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean
from collections import Counter
from game import Snake


class NeuralNet:
    def __init__(self, init_games=10000, test_games=500, goal_steps=5000, lr=1e-2, model_path='models/model'):
        self.init_games = init_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.model_path = model_path
        self.model = None
        self.vectors_and_keys = [
            [[-10, 0], 0],
            [[0, 10], 1],
            [[10, 0], 2],
            [[0, -10], 3]
        ]

    def initial_population(self):
        training_data = []
        for i in range(self.init_games):
            print("Training Game:", i + 1)
            game = Snake()
            _, prev_score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            prev_food_distance = self.get_food_distance(snake, food)
            for _ in range(self.goal_steps):
                action, game_action = self.generate_action(snake)
                done, score, snake, food = game.move(game_action)
                if done:
                    training_data.append([self.add_action_to_observation(prev_observation, action), -1])
                    # with open("training.txt", "a") as file:
                    #     file.write('Game' + str(i + 1) + '\t' + "  ----------  " + str(training_data[-1]) + "\n")
                    break
                else:
                    food_distance = self.get_food_distance(snake, food)
                    if score > prev_score or food_distance < prev_food_distance:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 1])
                    else:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 0])
                    prev_observation = self.generate_observation(snake, food)
                    prev_food_distance = food_distance
                # with open("training.txt", "a") as file:
                #     file.write('Game' + str(i+1)+'\t' + "  ----------  " + str(training_data[-1]) + "\n")
        return training_data

    def generate_observation(self, snake, food):
        snake_direction = self.get_snake_direction_vector(snake)
        food_direction = self.get_food_direction_vector(snake, food)
        barrier_left = self.is_direction_blocked(snake, self.turn_vector_to_the_left(snake_direction))
        barrier_front = self.is_direction_blocked(snake, snake_direction)
        barrier_right = self.is_direction_blocked(snake, self.turn_vector_to_the_right(snake_direction))
        angle = self.get_angle(snake_direction, food_direction)
        return np.array([int(barrier_left), int(barrier_front), int(barrier_right), angle])

    @staticmethod
    def get_snake_direction_vector(snake):
        return np.array(snake[0]) - np.array(snake[1])

    @staticmethod
    def get_food_direction_vector(snake, food):
        return np.array(food) - np.array(snake[0])

    @staticmethod
    def is_direction_blocked(snake, direction):
        point = np.array(snake[0]) + np.array(direction)
        return point.tolist() in snake[:] or point[0] == 0 or point[1] == 0 or point[0] == 501 or point[1] == 501

    @staticmethod
    def turn_vector_to_the_left(vector):
        return np.array([-vector[1], vector[0]])

    @staticmethod
    def turn_vector_to_the_right(vector):
        return np.array([vector[1], -vector[0]])

    @staticmethod
    def normalize_vector(vector):
        return vector / np.linalg.norm(vector)

    def get_food_distance(self, snake, food):
        return np.linalg.norm(self.get_food_direction_vector(snake, food))

    def generate_action(self, snake):
        action = randint(0, 2) - 1
        return action, self.get_game_action(snake, action)

    def get_game_action(self, snake, action):
        snake_direction = self.get_snake_direction_vector(snake)
        new_direction = snake_direction
        if action == -1:
            new_direction = self.turn_vector_to_the_left(snake_direction)
        elif action == 1:
            new_direction = self.turn_vector_to_the_right(snake_direction)
        for pair in self.vectors_and_keys:
            if pair[0] == new_direction.tolist():
                game_action = pair[1]
        return game_action

    def get_angle(self, a, b):
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)
        return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi

    @staticmethod
    def add_action_to_observation(observation, action):
        return np.append([action], observation)

    def get_model(self):
        network = input_data(shape=[None, 5], name='input')
        network = fully_connected(network, 32, activation='relu')
        network = fully_connected(network, 1, activation='linear')
        # acu = tflearn.metrics.Accuracy()
        network = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model

    def train_model(self, training_data):
        x = np.array([i[0] for i in training_data]).reshape(-1, 5)
        y = np.array([i[1] for i in training_data]).reshape(-1, 1)
        self.model.fit(x, y, n_epoch=3, shuffle=True, run_id=self.model_path)
        self.model.save(self.model_path)

    def _play(self, game, snake, prev_observation):
        predictions = []
        for action in range(-1, 2):
            predictions.append(
                self.model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5)))
        action = np.argmax(np.array(predictions))
        game_action = self.get_game_action(snake, action - 1)
        done, score, snake, food = game.move(game_action)
        return done, score, snake, food, action, predictions

    def test_model(self):
        steps_arr = []
        scores_arr = []
        for a in range(self.test_games):
            steps = 0
            game_memory = []
            game = Snake()
            _, score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            for _ in range(self.goal_steps):
                done, score, snake, food, action, predictions = self._play(game, snake, prev_observation)
                game_memory.append([prev_observation, action])
                if done:
                    result_string = steps + snake + food + prev_observation + predictions
                    print('-----')
                    print('Game Test', a)
                    print(result_string)
                    with open("testing.txt", "a") as file:
                        file.write(f'Test {a}\n {result_string}')
                    break
                else:
                    prev_observation = self.generate_observation(snake, food)
                    steps += 1
            steps_arr.append(steps)
            scores_arr.append(score)
        print('Average steps:', mean(steps_arr))
        print(Counter(steps_arr))
        print('Average score:', mean(scores_arr))
        print(Counter(scores_arr))
        step_str = str(mean(steps_arr))
        score_str = str(mean(scores_arr))
        with open("Stats.txt", "a") as file:
            file.write(step_str + "  ----------  " + score_str + "\n")

    def visualise_game(self):
        while True:
            game = Snake(is_gui=True)
            _, score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            for _ in range(self.goal_steps):
                done, score, snake, food, action, predictions = self._play(game, snake, prev_observation)
                if done:
                    break
                else:
                    prev_observation = self.generate_observation(snake, food)

    def train(self):
        training_data = self.initial_population()
        if self.model is None:
            self.model = self.get_model()
        self.train_model(training_data)

    def visualise(self):
        if self.model is None:
            self.model = self.get_model()
            self.model.load(self.model_path)
        self.visualise_game()

    def test(self):
        if self.model is None:
            self.model = self.get_model()
            self.model.load(self.model_path)
        self.test_model()


if __name__ == "__main__":
    neural_net = NeuralNet()
    neural_net.train()
    # neural_net.test()
    # neural_net.visualise()
