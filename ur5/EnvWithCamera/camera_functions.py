#%%
import numpy as np
import matplotlib.pyplot as plt

class CameraRail:
    
    
    def __init__(self, x_low, x_high, y_low, y_high, z_height, phi= 0): # phi in RAD
        x_mid = (x_high + x_low)/2
        y_mid = (y_high + y_low)/2
        self.center = np.array([x_mid, y_mid])
        self.radius = max((x_high - x_low)/2, (y_high - y_low)/2)
        self.z = z_height
        self.phi = phi
        self.position = self._get_coords()
        self.vel = 0


    def get_coords(self, d_phi, factor = 0.1):
        self.phi += np.clip(d_phi, -2*np.pi/50, 2*np.pi/50)
        
        return self._get_coords()

    def _get_coords(self) -> list:
        x = np.cos(self.phi) * self.radius
        y = np.sin(self.phi) * self.radius

        return [self.center[0] + x, self.center[1] + y, self.z]

#%%
if __name__ == '__main__':

    # theta goes from 0 to 2pi
    theta = np.linspace(0, 2*np.pi, 100)

    # the radius of the circle
    r = np.sqrt(1)

    # compute x1 and x2
    x1 = r*np.cos(theta)
    x2 = r*np.sin(theta)

    # create the figure
    fig, ax = plt.subplots(1)
    ax.plot(x1, x2)

    c = CameraRail(-1, 1, -0.1, 0.1, 0)
    point = c.get_coords(30)[:2]

    ax.plot(*point, 'x')

    ax.set_aspect(1)
    plt.show()
# %%
