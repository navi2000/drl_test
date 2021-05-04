import numpy as np
import tensorflow as tf
from Gridworld import Gridworld
import random
from matplotlib import pylab as plt
from collections import deque

LOAD_MODEL = False

action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r',
}

def test_model(model, mode='static', display=True):
    i = 0
    test_game = Gridworld(mode=mode)
    state_ = test_game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
    state = state_
    if display:
        print("Initial State:")
        print(test_game.display())
    status = 1
    while(status == 1): #A
        qval = model.predict(state)
        action_ = np.argmax(qval) #B
        action = action_set[action_]
        if display:
            print('Move #: %s; Taking action: %s' % (i, action))
        test_game.makeMove(action)
        state_ = test_game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
        state = state_
        if display:
            print(test_game.display())
        reward = test_game.reward()
        if reward != -1:
            if reward > 0:
                status = 2
                if display:
                    print("Game won! Reward: %s" % (reward,))
            else:
                status = 0
                if display:
                    print("Game LOST. Reward: %s" % (reward,))
        i += 1
        if (i > 15):
            if display:
                print("Game lost; too many moves.")
            break
    
    win = True if status == 2 else False
    return win


def get_model(load):
    if load:
        model = tf.keras.models.load_model("test_model.h5")
    else:
        model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(64)),
            tf.keras.layers.Dense(units=150, activation='relu'),
            tf.keras.layers.Dense(units=100, activation='relu'),
            tf.keras.layers.Dense(units=4, activation=None) 
        ])

    return model

model = get_model(LOAD_MODEL)
model2 = get_model(LOAD_MODEL)
model2.set_weights(model.get_weights())

loss_fn = tf.keras.losses.MSE
learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)

gamma = 0.9
epsilon = 0.3

epochs = 5000
losses = []
mem_size = 1000
batch_size = 200
replay = deque(maxlen=mem_size)
max_moves = 50
h = 0
sync_freq = 500 #A
j=0
for i in range(epochs):
    game = Gridworld(size=4, mode='random')
    state1_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
    state1 = state1_ #torch.from_numpy(state1_).float()
    status = 1
    mov = 0
    while(status == 1): 
        j+=1
        mov += 1
        qval = model.predict(state1)
        if (random.random() < epsilon):
            action_ = np.random.randint(0,4)
        else:
            action_ = np.argmax(qval)

        action = action_set[action_]
        game.makeMove(action)
        state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
        state2 = state2_ #torch.from_numpy(state2_).float()
        reward = game.reward()
        done = True if reward > 0 else False
        exp =  (state1, action_, reward, state2, done)
        replay.append(exp) #H
        state1 = state2

        if len(replay) > batch_size:
            minibatch = random.sample(replay, batch_size)
            state1_batch = np.array([s1 for (s1,a,r,s2,d) in minibatch]).reshape((batch_size, 64))
            action_batch = [a for (s1,a,r,s2,d) in minibatch]   #TODO: Posibles diferencies
            reward_batch = np.float32([r for (s1,a,r,s2,d) in minibatch])
            state2_batch = np.array([s2 for (s1,a,r,s2,d) in minibatch]).reshape((batch_size, 64))
            done_batch = np.array([d for (s1,a,r,s2,d) in minibatch])

            Q2 = model2(state2_batch) #B
            with tf.GradientTape() as tape:
                Q1 = model(state1_batch)
                Y = reward_batch + gamma * ((1-done_batch) * tf.math.reduce_max(Q2, axis=1))
                X = [Q1[i][action_batch[i]] for i in range(len(action_batch))]
                loss = loss_fn(X, Y)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            #loss = model.train_on_batch(X, Y)
            #print(i, loss)
            losses.append(loss)

            if j % sync_freq == 0: #C
                model2.set_weights(model.get_weights())
        if reward != -1 or mov > max_moves:
            status = 0
            mov = 0
        

plt.figure(figsize=(10,7))
plt.plot(losses)
plt.xlabel("Epochs",fontsize=22)
plt.ylabel("Loss",fontsize=22)
plt.show()

tf.keras.models.save_model(model, "tensorflow.h5")

max_games = 100
wins = 0
for i in range(max_games):
    win = test_model(model, mode='random', display=False)
    if win:
        wins += 1
win_perc = float(wins) / float(max_games)
print("Games played: {0}, # of wins: {1}".format(max_games,wins))
print("Win percentage: {}%".format(100.0*win_perc))


