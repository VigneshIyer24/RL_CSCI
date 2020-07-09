import numpy as np
import tensorflow as tf
import gym
import random
import matplotlib.pyplot as plt
from datetime import datetime
from buffer_mod import ReplayBuffer
from sys import exit

def Actor_NN(input_shape,num_action):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))    
    model.add(tf.keras.layers.Dense(units=1000, activation='relu'))
    model.add(tf.keras.layers.Dense(units=500, activation='relu'))
    model.add(tf.keras.layers.Dense(units=200, activation='relu'))
    model.add(tf.keras.layers.Dense(units=num_action, activation='tanh'))
    return model

def Critic_NN(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))    
    model.add(tf.keras.layers.Dense(units=400, activation='relu'))
    model.add(tf.keras.layers.Dense(units=200, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation=None))
    return model

def ddpg(env_fn,ac_kwargs=dict(),seed=0,save_folder='videos',num_train_episodes=200,test_agent_every=25,replay_size=int(1e6),
    gamma=0.99,decay=0.99,mu_lr=1e-4,q_lr=1e-4,batch_size=32,start_steps=1000,action_noise=random.uniform(0,1),max_episode_length=1000):

    tf.random.set_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()

    # get size of state space and action space
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    action_max = env.action_space.high[0]


   # Main network outputs
    mu = Actor_NN(num_states,num_actions)
    q_mu = Critic_NN((num_states+num_actions))

    # Target networks
    mu_target = Actor_NN(num_states,num_actions)
    q_mu_target = Critic_NN((num_states+num_actions))
 
    # Copying weights in,
    mu_target.set_weights(mu.get_weights())
    q_mu_target.set_weights(q_mu.get_weights())

    # Experience replay memory
    replay_buffer = ReplayBuffer(action_size=num_actions, buffer_size=replay_size,batch_size=batch_size,seed=seed)

    # Train each network separately
    mu_optimizer =tf.keras.optimizers.Adam(learning_rate=mu_lr)
    q_optimizer = tf.keras.optimizers.Adam(learning_rate=q_lr)
  
    def get_action(s, noise_scale):
        a = action_max * mu.predict(s.reshape(1,-1))[0]
        a += noise_scale * np.random.randn(num_actions)
        return np.clip(a, -action_max, action_max)
    
    test_returns = []
    def test_agent(num_episodes=200):
        t0 = datetime.now()
        n_steps = 0
        for j in range(num_episodes):
            s, episode_return, episode_length, d = test_env.reset(), 0, 0, False
            while not (d or (episode_length == max_episode_length)):
                # Take deterministic actions at test time (noise_scale=0)
                test_env.render()
                s, r, d, _ = test_env.step(get_action(s, 0))
                episode_return += r
                episode_length += 1
                n_steps += 1
            print('episode number:', j, 'test return:', episode_return, 'episode_length:', episode_length)
            test_returns.append(episode_return)  
    # Main loop: play episode and train
    returns = []
    q_losses = []
    mu_losses = []
    num_steps = 0
    for i_episode in range(num_train_episodes):
        state=env.reset()
        episode_reward=0
        episode_length = 0
        while not (episode_length == max_episode_length):
            if num_steps > start_steps:
                action = get_action(state, action_noise)
            else:
                action = env.action_space.sample()

            # Keep track of the number of steps done
            num_steps += 1
            if num_steps == start_steps:
                print("USING AGENT ACTIONS NOW")
            action = get_action(state, action_noise)
            # Step the env
            next_state, reward, d, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            d_store = False if episode_length == max_episode_length else d
            # Store experience to replay buffer
            replay_buffer.add(state,action,reward,next_state,d_store)
            # Assign next state to be the current state on the next round
            state = next_state
                        

        # Perform the updates
        for _ in range(episode_length):
      
            X,A,R,X2,D = replay_buffer.sample_determination(batch_size)
            
            #Critic Optimization
            with tf.GradientTape() as tape:
                next_a = action_max * mu_target(X2)
                temp = np.concatenate((X2,next_a),axis=1)
                q_target = R + gamma * (1 - D) * q_mu_target(temp)
                temp2 = np.concatenate((X,A),axis=1)
                qvals = q_mu(temp2) 
                q_loss = tf.reduce_mean((qvals - q_target)**2)
                grads_q = tape.gradient(q_loss,q_mu.trainable_variables)
            q_optimizer.apply_gradients(zip(grads_q, q_mu.trainable_variables))
            q_losses.append(q_loss)
            
            
            #Actor optimization   
            with tf.GradientTape() as tape2:
                Aprime = action_max * mu(X)
                temp = tf.keras.layers.concatenate([X,Aprime],axis=1)
                Q = q_mu(temp)
                mu_loss =  -tf.reduce_mean(Q)
                grads_mu = tape2.gradient(mu_loss,mu.trainable_variables)
            mu_losses.append(mu_loss)
            mu_optimizer.apply_gradients(zip(grads_mu, mu.trainable_variables))
      
            
      
            temp1 = np.array(q_mu_target.get_weights())
            temp2 = np.array(q_mu.get_weights())
            temp3 = decay*temp1 + (1-decay)*temp2
            q_mu_target.set_weights(temp3)
     

            # updating Actor network
            temp1 = np.array(mu_target.get_weights())
            temp2 = np.array(mu.get_weights())
            temp3 = decay*temp1 + (1-decay)*temp2
            mu_target.set_weights(temp3)
      

        print("Episode:", i_episode + 1, "Return:", episode_reward, 'episode_length:', episode_length)
        returns.append(episode_reward)
    test_agent()
    return (returns,q_losses,mu_losses,test_returns)