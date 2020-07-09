{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Episode: 1 Return: -75.53994801491808 episode_length: 250\n",
      "Episode: 2 Return: -70.43004916510145 episode_length: 250\n",
      "Episode: 3 Return: 9.380157424176694 episode_length: 250\n",
      "USING AGENT ACTIONS NOW\n",
      "Episode: 4 Return: -117.3364756343666 episode_length: 250\n",
      "Episode: 5 Return: -80.76612247145259 episode_length: 250\n",
      "Episode: 6 Return: -89.46122875066571 episode_length: 250\n",
      "Episode: 7 Return: -70.28166675478644 episode_length: 250\n",
      "Episode: 8 Return: -83.318830453255 episode_length: 250\n",
      "Episode: 9 Return: -107.32914286903214 episode_length: 250\n",
      "Episode: 10 Return: -75.17095266224528 episode_length: 250\n",
      "Episode: 11 Return: -100.87954243147394 episode_length: 250\n",
      "Episode: 12 Return: -110.64866639616699 episode_length: 250\n",
      "Episode: 13 Return: -109.43329017984301 episode_length: 250\n",
      "Episode: 14 Return: -67.86910165103156 episode_length: 250\n"
     ]
    }
   ],
   "source": [
    "import gym\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from ddpg_mod import ddpg\n",
    "\n",
    "def smooth(x):\n",
    "    n = len(x)\n",
    "    y = np.zeros(n)\n",
    "    for i in range(n):\n",
    "        start = max(0, i - 99)\n",
    "        y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)\n",
    "    return y\n",
    "returns, q_losses,mu_losses,test_returns = ddpg(lambda : gym.make('HalfCheetah-v2'),num_train_episodes=50)\n",
    "\n",
    "\n",
    "plt.plot(returns)\n",
    "plt.plot(smooth(np.array(returns)))\n",
    "plt.title(\"Train returns\")\n",
    "plt.show()\n",
    "\n",
    "plt.plot(q_losses)\n",
    "plt.title('q_losses')\n",
    "plt.show()\n",
    "\n",
    "plt.plot(mu_losses)\n",
    "plt.title('mu_losses')\n",
    "plt.show()\n",
    "\n",
    "plt.plot(test_returns)\n",
    "plt.plot(smooth(np.array(test_returns)))\n",
    "plt.title('test_rewards')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
