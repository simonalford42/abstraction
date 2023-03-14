import time
import numpy as np
import matplotlib
import box_world
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def save_image(state, file_name):
    color_array = box_world.to_color_obs(state)
    plt.imshow(color_array / 255)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'/Users/simon/Desktop/{file_name}.png')
    plt.close()


def save_video(video_obss, file_name):
    # convert pauses
    fps = 5
    # repeat frames
    video_obss2 = []
    for (obs, title, pause) in video_obss:
        for i in range(int(pause * fps)):
            video_obss2.append((obs, title))

    fig = plt.figure()

    def updatefig(video_obs):
        obs, title = video_obs
        color_array = box_world.to_color_obs(obs)
        im = plt.imshow(color_array / 255)
        if title:
            plt.title(title)
        plt.xticks([])
        plt.yticks([])
        return im,

    ani = animation.FuncAnimation(fig, updatefig, frames=video_obss2, blit=True, repeat=False)
    ani.save(f'/Users/simon/Desktop/{file_name}.gif', writer='imagemagick', fps=fps)
