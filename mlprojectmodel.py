import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class SmartIrrigationEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # Actions:
        # 0 = Pump OFF
        # 1 = LOW Water
        # 2 = MEDIUM Water
        # 3 = HIGH Water
        self.action_space = spaces.Discrete(4)

        # Observation Space
        # [soil, temperature, humidity, rain, tank]
        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(5,),
            dtype=np.float32
        )

        self.state = None

        # Pygame variables
        self.window = None
        self.clock = None

        self.window_width = 800
        self.window_height = 500

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.state = np.array(
            [50, 30, 60, 20, 80],
            dtype=np.float32
        )

        return self.state, {}

    def step(self, action):

        soil, temp, humidity, rain, tank = self.state

        # Pump water effect
        soil += action * 5

        # Natural drying
        soil -= np.random.uniform(1, 3)

        # Rain effect
        if rain > 70:
            soil += 4

        # Clamp values
        soil = np.clip(soil, 0, 100)

        # Reward system
        reward = 0

        if 40 <= soil <= 70:
            reward += 10

        if soil > 90:
            reward -= 20

        # Water saving penalty
        reward -= action * 2

        # Update state
        self.state = np.array(
            [soil, temp, humidity, rain, tank],
            dtype=np.float32
        )

        terminated = False
        truncated = False

        return self.state, reward, terminated, truncated, {}

    def render(self):

        if self.render_mode != "human":
            return

        # Initialize pygame
        if self.window is None:

            pygame.init()

            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )

            pygame.display.set_caption(
                "Smart Irrigation Controller"
            )

            self.clock = pygame.time.Clock()

        # Background color
        self.window.fill((240, 240, 240))

        soil = float(self.state[0])

        # Title
        font = pygame.font.Font(None, 48)

        title = font.render(
            "Smart Irrigation System",
            True,
            (0, 100, 0)
        )

        self.window.blit(title, (200, 30))

        # Draw moisture bar outline
        pygame.draw.rect(
            self.window,
            (0, 0, 0),
            (150, 150, 500, 50),
            3
        )

        # Moisture fill bar
        pygame.draw.rect(
            self.window,
            (0, 0, 255),
            (150, 150, int(soil * 5), 50)
        )

        # Soil moisture text
        font2 = pygame.font.Font(None, 36)

        moisture_text = font2.render(
            f"Soil Moisture: {soil:.1f}%",
            True,
            (0, 0, 0)
        )

        self.window.blit(moisture_text, (260, 230))

        # Pump status
        if soil < 40:
            status = "Pump ON"
            color = (255, 0, 0)
        else:
            status = "Pump OFF"
            color = (0, 150, 0)

        status_text = font2.render(
            status,
            True,
            color
        )

        self.window.blit(status_text, (330, 300))

        # Update screen
        pygame.display.update()

        # FPS
        self.clock.tick(2)

    def close(self):

        if self.window is not None:
            pygame.quit()


# Create environment
env = SmartIrrigationEnv(render_mode="human")

# Reset environment
state, info = env.reset()

# Run simulation
for step in range(50):

    # Random action
    action = env.action_space.sample()

    # Apply action
    state, reward, terminated, truncated, info = env.step(action)

    print(f"Step: {step + 1}")
    print("Action:", action)
    print("State:", state)
    print("Reward:", reward)
    print("---------------------")

    # Render window
    env.render()

# Close environment
env.close()
