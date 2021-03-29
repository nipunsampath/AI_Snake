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
    def __init__(self, init_games=10000, test_games=500, goal_steps=5000, lr=1e-2, filename='model'):
        self.init_games = init_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.filename = filename
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
            self.board = game.board
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

    def get_snake_direction_vector(self, snake):
        return np.array(snake[0]) - np.array(snake[1])

    def get_food_direction_vector(self, snake, food):
        return np.array(food) - np.array(snake[0])

    def is_direction_blocked(self, snake, direction):
        point = np.array(snake[0]) + np.array(direction)
        return point.tolist() in snake[:] or point[0] == 0 or point[1] == 0 or point[0] == 501 or point[1] == 501

    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], -vector[0]])

    def normalize_vector(self, vector):
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

    def add_action_to_observation(self, observation, action):
        return np.append([action], observation)

    def model(self):
        network = input_data(shape=[None, 5, 1], name='input')
        network = fully_connected(network, 32, activation='relu')
        network = fully_connected(network, 1, activation='linear')
        acu = tflearn.metrics.Accuracy()
        network = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model

    def train_model(self, training_data, model):
        X = np.array([i[0] for i in training_data]).reshape(-1, 5, 1)
        y = np.array([i[1] for i in training_data]).reshape(-1, 1)
        model.fit(X, y, n_epoch=3, shuffle=True, run_id=self.filename)
        model.save(self.filename)
        return model

    def test_model(self, model):
        steps_arr = []
        scores_arr = []
        for a in range(self.test_games):
            steps = 0
            game_memory = []
            game = Snake()
            _, score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            for _ in range(self.goal_steps):
                predictions = []
                for action in range(-1, 2):
                    predictions.append(
                        model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
                action = np.argmax(np.array(predictions))
                game_action = self.get_game_action(snake, action - 1)
                done, score, snake, food = game.move(game_action)
                game_memory.append([prev_observation, action])
                if done:
                    result_string = steps + snake + food + prev_observation + predictions
                    print('-----')
                    print('Game Test', a)
                    # print(steps)
                    # print(snake)
                    # print(food)
                    # print(prev_observation)
                    # print(predictions)
                    print(result_string)
                    with open("testing.txt", "a") as file:
                        file.write('Test ' + a + '\n' + result_string)
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

    def visualise_game(self, model):
        while True:
            game = Snake(is_gui=True)
            _, score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            for _ in range(self.goal_steps):
                predictions = []
                for action in range(-1, 2):
                    predictions.append(
                        model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
                action = np.argmax(np.array(predictions))
                game_action = self.get_game_action(snake, action - 1)
                done, _, snake, food = game.move(game_action)

                if done:
                    break
                else:
                    prev_observation = self.generate_observation(snake, food)

    def train(self):
        training_data = self.initial_population()
        nn_model = self.model()
        nn_model = self.train_model(training_data, nn_model)

    def visualise(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.visualise_game(nn_model)

    def test(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.test_model(nn_model)


if __name__ == "__main__":
    NeuralNet().train()
    #NeuralNet().test()
    #NeuralNet().visualise()
