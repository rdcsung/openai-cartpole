import gym
import numpy as np
import matplotlib.pyplot as plt


def run_episode(env, parameters, total_steps):
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        total_steps += 1
        if done:
            break
    return totalreward, total_steps

def train(exp, submit):
    env = gym.make('CartPole-v0')

    counter = 0
    bestparams = None
    bestreward = 0
    total_steps = 0

    for _ in range(10000):
        counter += 1
        parameters = np.random.rand(4) * 2 - 1
        reward , total_steps = run_episode(env,parameters, total_steps)
        print("exp:{} epsode:{}, total steps: {}, rewards: {}".
              format(exp, counter, total_steps, reward))
        if reward > bestreward:
            bestreward = reward
            bestparams = parameters
            if reward == 200:
                break

    if submit:
        rewards = []
        total_steps = 0
        for i in range(100):
            reward, total_steps = run_episode(env, bestparams, total_steps)
            rewards.append(reward)
            # print("=====Exp:{}, Test round:{},reward:{}".format(exp, i, reward))

        print("+++++Exp:{}, Average reward reward:{}".format(exp, np.mean(rewards)))


    return counter, total_steps

# train an agent to submit to openai gym
# train(submit=True)

# create graphs
eps = []
steps = []
for exp in range(1000):
    ep, step = train(exp, submit=True)
    eps.append(ep)
    steps.append(step)

'''
plt.hist(eps,50,normed=1, facecolor='g', alpha=0.75)
plt.xlabel('Episodes required to reach 200')
plt.ylabel('Frequency')
plt.title('Histogram of Random Search')
plt.show()

plt.hist(steps,50,normed=1, facecolor='g', alpha=0.75)
plt.xlabel('Steps required to reach 200')
plt.ylabel('Frequency')
plt.title('Histogram of Random Search')
plt.show()
'''

print("Average episodes to solve the problme: {}".format(np.mean(eps)))

print("Average steps to solve the problme: {}".format(np.mean(steps)))