import pygame

class KeybordControllerClass:
    def __init__(self):
        pygame.init()
        pygame.font.init()  # you have to call this at the start,
        self.font_size = 30
        self.my_font = pygame.font.SysFont('Comic Sans MS', self.font_size)
        self.win = pygame.display.set_mode((400,400))

    def drawWarning(self, value, text=[]):
        # Initialing Color
        self.win.fill(pygame.Color('black'))
        if value == True:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        # Drawing Rectangle
        pygame.draw.rect(self.win, color, pygame.Rect(0, 200, 400, 200))
        yy = 0
        for tt in text:
            text_surface = self.my_font.render(tt, False, (255, 255, 255))
            self.win.blit(text_surface, (0, yy))
            yy += self.font_size + 1
        #pygame.display.flip()
        pygame.display.update()


    def getKey(self, keyPressed):
        ans = False
        for eve in pygame.event.get(): pass
        keyInput = pygame.key.get_pressed()
        myKey = getattr(pygame, 'K_{}'.format(keyPressed))
        if keyInput[myKey]:
            ans = True
        pygame.display.update()
        return ans

    """
    def main():
        print(getKey("a"))
    """

"""
if __name__ == '__main__':
    init()
    while True:
        main()

"""