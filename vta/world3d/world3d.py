import enum
import gym
import gym_miniworld
import numpy as np
from gym_miniworld import miniworld
import abc
from PIL import Image, ImageDraw


class MultiTask3DEnv(miniworld.MiniWorldEnv):
    """3D room filled with objects.

    The agent must pick up the correct objects in the correct order to solve the
    task.
    """

    def __init__(self, size=10, max_episode_steps=100, num_objects=6,
                 seed=0, visit_length=3):
        assert num_objects <= len(gym_miniworld.entity.COLORS)
        params = gym_miniworld.params.DEFAULT_PARAMS.no_random()
        params.set('forward_step', 0.7)
        params.set('turn_step', 45)  # 45 degree rotation

        self._num_objects = num_objects
        self._visit_length = visit_length
        self._size = size
        self._visit_index = 0
        self.seed(seed)

        super().__init__(
                params=params, max_episode_steps=max_episode_steps,
                domain_rand=False)

        # Allow for left / right / forward
        self.action_space = gym.spaces.Discrete(self.actions.move_forward + 1)

    def seed(self, seed=None):
        rand_gen = super().seed(seed)
        self._colors = self.rand.subset(
                gym_miniworld.entity.COLOR_NAMES, self._num_objects)

        def min_distance(pos, positions):
            return min(np.square(np.linalg.norm(
                np.array(pos) - np.array(pos2), ord=2))
                for pos2 in positions)

        self._positions = [(self.rand.float(2, self._size - 2), 0.5,
                            self.rand.float(2, self._size - 2))]
        for _ in range(self._num_objects - 1):
            while True:
                new_pos = (self.rand.float(2, self._size - 2), 0.5,
                           self.rand.float(2, self._size - 2))
                if min_distance(new_pos, self._positions) > 2:
                    break

            self._positions.append(new_pos)

        self._items_to_visit = self.rand.subset(
                self._colors, self._visit_length)

        return rand_gen

    def _gen_world(self):
        self.add_rect_room(min_x=0, max_x=self._size, min_z=0, max_z=self._size)

        self._objects = [
            self.place_entity(gym_miniworld.entity.Box(color=color), pos=pos)
            for color, pos in zip(self._colors, self._positions)]

        self.place_agent(min_x=0, max_x=self._size, min_z=0, max_z=self._size)

    def reset(self):
        self._visit_index = 0
        state = super().reset()
        return state, self._items_to_visit[self._visit_index]

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if action == self.actions.pickup and self.agent.carrying:
            if self.agent.carrying.color == self.goal_color:
                reward = 1
                self._visit_index += 1

            self.entities.remove(self.agent.carrying)
            # Don't bother storing picked up objects in the inventory
            self.agent.carrying = None
            obs = self.render_obs()
        return (obs, self.goal_color), reward, done, info

    @property
    def goal_color(self):
        # Random color if task is already done
        if self._visit_index >= len(self._items_to_visit):
            return self._items_to_visit[0]
        return self._items_to_visit[self._visit_index]

class PanoramaObservationWrapper(gym.Wrapper):
    """Returns observations that are 4 90 degree camera angles."""

    def gen_panorama_obs(self, front_view):
        views = [front_view]
        for _ in range(3):
            self.env.turn_agent(90)
            view.append(self.env.render_obs())
        self.env.turn_agent(90)
        return np.concatenate(views, -1)

    def step(self, action):
        front_view, reward, done, info = super().step(action)
        return self.gen_panorama_obs(front_view), reward, done, info

    def reset(self):
        return self.gen_panorama_obs(super().reset())


class FourDirectionActionWrapper(gym.Wrapper):
    """Provides a new set of actions for interacting with the environment."""

    class Actions(enum.IntEnum):
        up = 0
        down = 1
        left = 2
        right = 3
        pickup = 4

    def step(self, action):
        if action == self.actions.up:
            return self.env.step(self.env.actions.move_forward)
        elif action == self.actions.down:
            return self.env.step(self.env.actions.move_back)
        elif action == self.actions.left:
            self.env.turn_agent(90)
            state, reward, done, info = self.env.step(self.env.actions.move_forward)
            self.env.turn_agent(-90)
            return state, reward, done, info
        elif action == self.actions.right:
            self.env.turn_agent(90)
            state, reward, done, info = self.env.step(self.env.actions.move_back)
            self.env.turn_agent(-90)
            return state, reward, done, info
        elif action == pickup:
            return self.env.step(self.actions.pickup)
        else:
            raise ValueError()

    def reset(self):
        obs = super().reset()
        self.env.turn_agent(-self.env.agent.dir)
        return self.env.render_obs()


# From DREAM
class Render(abc.ABC):
  """Convenience object to return from env.render.
  Allows for annotated text on a banner on top and exporting to PIL.
  """

  def __init__(self, main_image):
    """Constructs from the main PIL image to render.
    Args:
      main_image (PIL.Image): the main thing to render.
    """
    self._main_image = Image.fromarray(main_image.astype(np.uint8)).resize((800, 600))
    self._banner = Image.new(
        mode="RGBA", size=(self._main_image.width, 150), color="white")
    self._text = []

  def write_text(self, text):
    """Appends new line of text to any previous text.
    Args:
      text (str): text to display on top of rendering.
    """
    self._text.append(text)

  def image(self):
    """Returns a PIL.Image representation of this rendering."""
    draw = ImageDraw.Draw(self._banner)
    draw.text((0, 0), "\n".join(self._text), (0, 0, 0))
    return concatenate([self._banner, self._main_image], "vertical")

  def __deepcopy__(self, memo):
    cls = self.__class__
    deepcopy = cls.__new__(cls)
    memo[id(self)] = deepcopy

    # PIL doesn't support deepcopy directly
    image_copy = self.__dict__["_main_image"].copy()
    banner_copy = self.__dict__["_banner"].copy()
    setattr(deepcopy, "_main_image", image_copy)
    setattr(deepcopy, "_banner", banner_copy)

    for k, v in self.__dict__.items():
      if k not in ["_main_image", "_banner"]:
        setattr(deepcopy, k, copy.deepcopy(v, memo))
    return deepcopy


def concatenate(images, mode="horizontal"):
  assert mode in ["horizontal", "vertical"]

  if mode == "horizontal":
    new_width = sum(img.width for img in images)
    new_height = max(img.height for img in images)
  else:
    new_width = max(img.width for img in images)
    new_height = sum(img.height for img in images)

  final_image = Image.new(mode="RGBA", size=(new_width, new_height))
  curr_width, curr_height = (0, 0)
  for img in images:
    final_image.paste(img, (curr_width, curr_height))

    if mode == "horizontal":
      curr_width += img.width
    else:
      curr_height += img.height
  return final_image
