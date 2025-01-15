import pygame
import numpy as np
import hockey.hockey_env as h_env

def main():
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("Manual Hockey Play")

    env = h_env.HockeyEnv()  # Consider env = h_env.HockeyEnv(render_mode="human") if needed
    obs, info = env.reset()
    
    action_player1 = np.array([0.0, 0.0, 0.0, 0.0])
    action_player2 = np.array([0.0, 0.0, 0.0, 0.0])

    clock = pygame.time.Clock()
    running = True

    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action_player1[1] = +1.0
                elif event.key == pygame.K_DOWN:
                    action_player1[1] = -1.0
            if event.type == pygame.KEYUP:
                if event.key in (pygame.K_UP, pygame.K_DOWN):
                    action_player1[1] = 0.0

        # Always step, but ignore the environment's done to keep it running
        obs, reward, terminated, truncated, info = env.step(np.hstack([action_player1, action_player2]))
        env.render()

        # Comment out or remove these lines to prevent auto-close:
        # if terminated or truncated:
        #     running = False

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()