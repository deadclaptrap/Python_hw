import numpy as np
import imageio
from vispy import app, scene
from vispy.geometry import Rect
from numba import jit, prange, njit

# from funcs import init_boids, directions, propagate, flocking
app.use_app('pyglet')
w, h = 1280, 960
N = 5000
dt = 0.1
asp = w / h
perception = 0.01
# walls_order = 8
better_walls_w = 0.05
vrange = (0, 0.01)
arange = (0, 0.005)

#                    c      a    s      w

cord = np.array([[0.5, 0.8, 0.05], [0.3, 0.5, 0.05], [0.1, 0.2, 0.1], [0.5, 0.1, 0.06],
                     [0.7, 0.2, 0.06], [1, 0.6, 0.2]])

def init_boids(boids: np.ndarray, asp: float, vrange: tuple[float, float]):
    """
    initialize the boids array with random numbers
    """
    n = boids.shape[0]
    rng = np.random.default_rng()
    boids[:, 0] = rng.uniform(0., asp, size=n)
    boids[:, 1] = rng.uniform(0., 1., size=n)
    alpha = rng.uniform(0, 2 * np.pi, size=n)
    v = rng.uniform(*vrange, size=n)
    c, s = np.cos(alpha), np.sin(alpha)
    boids[:, 2] = v * c
    boids[:, 3] = v * s


@njit()
def directions(boids: np.ndarray, dt: float) -> np.ndarray:
    """
    gets the direction of boids to draw arrows
    :param boids:
    :param dt:
    :return: array N x (x0, y0, x1, y1) for arrow painting
    """
    return np.hstack((
        boids[:, :2] - dt * boids[:, 2:4],
        boids[:, :2]
    ))
@njit()
def norma(v: np.ndarray, num: int):
    """"
    norm of vector
    num - dimensions
    """
    return np.sqrt(np.sum(v**2, axis=num))

@njit()
def vclip(v: np.ndarray, vrange: tuple[float, float]):
    """"
    normalizations of boids(speed and acceleration) based on range
    v - boids speed
    vrange - range of speed
    """
    #norm = np.linalg.norm(v, axis=1)
    norm = norma(v, 1)
    mask = norm > vrange[1]
    if np.any(mask):
        v[mask] *= (vrange[1] / norm[mask]).reshape(-1, 1)


def propagate(boids: np.ndarray,
              dt: float,
              vrange: tuple[float, float],
              arange: tuple[float, float]):
    """"
    updates the boids
    """
    vclip(boids[:, 4:6], arange)
    boids[:, 2:4] += dt * boids[:, 4:6]
    vclip(boids[:, 2:4], vrange)
    boids[:, 0:2] += dt * boids[:, 2:4]


D = np.zeros((N, N))

@njit()
def distances(vecs: np.ndarray)-> np.ndarray:
    """"
    returns the array of distances between boids
    vecs = boids
    """
    n, m = vecs.shape
    vecs = vecs.copy()
    delta = vecs.reshape((n, 1, m)) - vecs.reshape((1, n, m))

    d = norma(delta, 2)
    return d

@njit()
def distances_walls(boidsc: np.ndarray, obj, preseption) -> np.ndarray:
    """"
    returns the array of distances between boids and objects
    boidsc = array of boids
    obj = array of objects
    """
    n, m = boidsc.shape
    n1, m1 = obj.shape
    boidsc, obj = boidsc.copy(), obj.copy()
    D = np.zeros((n, n1, 2))
    D1 = np.zeros((n, n1))
    for i in prange(n):
        for j in prange(n1):
            D[i, j, 0] = boidsc[i, 0] - obj[j, 0]
            D[i, j, 1] = boidsc[i, 1] - obj[j, 1]
            # print(D[i, j])
            D1[i, j] = np.linalg.norm(D[i, j]) - obj[j, 2]
            # if (D1[i, j] < preseption):
            # D1[i, j] = 1
            # r = 0
            # else:
            # D1[i, j] = 0
    return D1 < (perception)

@njit()
def cohesion(boids: np.ndarray,
             idx: int,
             neigh_mask: np.ndarray,
             perception: float) -> np.ndarray:

    """"
    returns the acceleration vector created by steer to move toward the average position of local boids
    boids - array of boids
    neigh_mask - array of boids that are close enough to boid [idx]
    """
    center = mean(boids[neigh_mask, :2], 0)
    a = (center - boids[idx, :2]) / perception
    return a

@njit()
def separation(boids: np.ndarray,
               idx: int,
               neigh_mask: np.ndarray,
               perception: float) -> np.ndarray:
    """"
    returns the separation between boid with number idx and other boids
    retuns array[2] - the acceleration
    boids - array of boids
    neigh_mask - array of boids that are close enough to boid [idx]
    """
    neighbs = boids[neigh_mask, :2] - boids[idx, :2]
    # print(neighbs)
    norm = norma(neighbs, 1)
    #norm = np.linalg.norm(neighbs, axis=1)
    mask = norm > 0
    # print(mask)
    # print("")
    if np.any(mask):
        neighbs[mask] /= norm[mask].reshape(-1, 1)
    d = mean(neighbs, 0) #.mean(axis=0)
    norm_d = norma(d, 0) #np.linalg.norm(d)
    if norm_d > 0:
        d /= norm_d
    # d = (boids[neigh_mask, :2] - boids[idx, :2]).mean(axis=0)
    return -d  # / ((d[0] ** 2 + d[1] ** 2) + 1)

@njit()
def mean(v: np.ndarray, t):

    """
    mean of vector
    t - dimensions
    """
    if t == 0:
        res = np.empty(v.shape[1], dtype=float)
        for i in range(len(res)):
            res[i] = np.mean(v[:, i])
    else:
        res = np.empty(v.shape[0], dtype=float)
        for i in range(len(res)):
            res[i] = np.mean(v[i, :])
    return res

@njit()
def alignment(boids: np.ndarray,
              idx: int,
              neigh_mask: np.ndarray,
              vrange: tuple) -> np.ndarray:
    """"
    returns the acceleration vector created by steer towards the average heading of local boids
    """
    v_mean = mean(boids[neigh_mask, 2:4], 0)
    a = (v_mean - boids[idx, 2:4]) / (2 * vrange[1])
    return a
""""

def walls(boids: np.ndarray, asp: float, param: int):
   
    c = 1
    x = boids[:, 0]
    y = boids[:, 1]
    order = param

    a_left = 1 / (np.abs(x) + c) ** order
    a_right = -1 / (np.abs(x - asp) + c) ** order

    a_bottom = 1 / (np.abs(y) + c) ** order
    a_top = -1 / (np.abs(y - 1.) + c) ** order

    return np.column_stack((a_left + a_right, a_bottom + a_top))
"""
@njit()
def smoothstep(edge0: float, edge1: float, x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip((x - edge0) / (edge1 - edge0), 0., 1.)
    return x * x * (3.0 - 2.0 * x)

@njit()
def better_walls(boids: np.ndarray, asp: float, param: float):
    """"
    returns position of walls
    """
    x = boids[:, 0]
    y = boids[:, 1]
    w = param

    a_left = smoothstep(asp * w, 0.0, x)
    a_right = -smoothstep(asp * (1.0 - w), asp, x)

    a_bottom = smoothstep(w, 0.0, y)
    a_top = -smoothstep(1.0 - w, 1.0, y)

    return np.column_stack((a_left + a_right, a_bottom + a_top))

@njit()
def separation_of_walls_(boids: np.ndarray,
                         idx: int,
                         neigh_mask: np.ndarray,
                         perception: float) -> np.ndarray:
    """"
    returns the separation between boid with number idx and objects
    retuns array[2] - the acceleration
    """


    neighbs = cord[neigh_mask, :2] - boids[idx, :2]
    norm = norma(neighbs, 1)
    #norm = np.linalg.norm(neighbs, axis=1)
    mask = norm > 0
    if np.any(mask):
        neighbs[mask] /= norm[mask].reshape(-1, 1)
    #d = neighbs.mean(axis=0)
    d = mean(neighbs, 0)
    norm_d = np.linalg.norm(d)
    if norm_d > 0:
        d /= norm_d
    return -d

@njit()
def mask_(boids: np.ndarray,
               perception: float):
    """"
    returns mask of distanses between boids
    """
    arr = distances(boids[:, :2])
    np.fill_diagonal(arr, perception + 1)
    return arr < perception

@njit()
def noise():
    """Generates noise in (-1, 1) interval """
    arr = np.random.rand(2)
    if np.random.rand(1) > .5:
        arr[0] *= -1
    if np.random.rand(1) > .5:
        arr[1] *= -1
    return arr

@njit(parallel=True)
def flocking(boids: np.ndarray,
             perception: float,
             coeffs: np.ndarray,
             asp: float,
             vrange: tuple,
             order: int):
    """"
    changes the boids array usind separation, cohesion ect
    """
    D = mask = mask_(boids, perception)#distances(boids[:, 0:2], perception)
    obj = np.array([[0.5, 0.8, 0.05], [0.3, 0.5, 0.05], [0.1, 0.2, 0.1], [0.5, 0.1, 0.06],
                    [0.7, 0.2, 0.06], [1, 0.6, 0.2]])
    mask_walls = distances_walls(boids[:, 0:2], obj, perception)
    N = boids.shape[0]
    #D[range(N), range(N)] = perception + 1
    #mask = D < perception
    #mask_walls = D1 < (perception)
    wal = better_walls(boids, asp, order)
    for i in prange(N):
        if not np.any(mask_walls[i]):
            circ = np.zeros(2)
        else:
            circ = separation_of_walls_(boids, i, mask_walls[i], perception)
        if not np.any(mask[i]):

            coh = np.zeros(2)
            alg = np.zeros(2)
            sep = np.zeros(2)
            nois = np.zeros(2)
        else:
            coh = cohesion(boids, i, mask[i], perception)
            alg = alignment(boids, i, mask[i], vrange)
            sep = separation(boids, i, mask[i], perception)
            nois = noise()
        a = coeffs[0] * (coh) + coeffs[1] * (alg) + \
            coeffs[2] * (sep + circ) + coeffs[3] * (wal[i])
        a += nois*0.01
        #print("n: ", nois)
        #print(a)
        boids[i, 4:6] = a


w, h = 1280, 960
N = 5000
dt = 0.1
asp = w / h
perception = 0.01
# walls_order = 8
better_walls_w = 0.05
vrange = (0, 0.01)
arange = (0, 0.005)

#                    c      a    s      w
#coeffs = np.array([0.05, 0.02, 0.1, 0.05])
coeffs = np.array([0.4, 1.2, 0.25, 0.8])
# 0  1   2   3   4   5
# x, y, vx, vy, ax, ay
boids = np.zeros((N, 6), dtype=np.float64)
init_boids(boids, asp, vrange=vrange)
# boids[:, 4:6] = 0.1

canvas = scene.SceneCanvas(show=True, size=(w, h))
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=Rect(0, 0, asp, 1))
arrows = scene.Arrow(arrows=directions(boids, dt),
                     arrow_color=(1, 1, 1, 1),
                     # width=5,
                     arrow_size=5,
                     connect='segments',
                     parent=view.scene)

ellipse = scene.visuals.Ellipse(center=(0.5, 0.8), radius=(0.05, 0.05),
                                color='#FF000000',
                                border_color=(1, 1, 1),
                                parent=view.scene)
ellipse = scene.visuals.Ellipse(center=(0.3, 0.5), radius=(0.05, 0.05),
                                color='#FF000000',
                                border_color=(1, 1, 1),
                                parent=view.scene)
ellipse = scene.visuals.Ellipse(center=(0.1, 0.2), radius=(0.1, 0.1),
                                color='#FF000000',
                                border_color=(1, 1, 1),
                                parent=view.scene)
ellipse = scene.visuals.Ellipse(center=(0.5, 0.1), radius=(0.06, 0.06),
                                color='#FF000000',
                                border_color=(1, 1, 1),
                                parent=view.scene)
ellipse = scene.visuals.Ellipse(center=(1, 0.6), radius=(0.2, 0.2),
                                color='#FF000000',
                                border_color=(1, 1, 1),
                                parent=view.scene)
ellipse = scene.visuals.Ellipse(center=(0.7, 0.2), radius=(0.06, 0.06),
                                color='#FF000000',
                                border_color=(1, 1, 1),
                                parent=view.scene)
# [0.5, 0.8, 0.1], [0.7, 0.2, 0.06] , [1, 0.6, 0.2]
# w = imageio.get_writer('my_video.mp4', format='FFMPEG', mode='I', fps=1)
txt = scene.visuals.Text(parent=canvas.scene, color='white')
txt.pos = 15 * canvas.size[0] // 16, canvas.size[1] // 35
txt.font_size = 10
txt_const = scene.visuals.Text(parent=canvas.scene, color='white')
txt_const.pos = canvas.size[0] // 16, canvas.size[1] // 10
txt_const.font_size = 8

writer = imageio.get_writer(f'birds{5000}.mp4', fps=60)
fr = 0

def update(event):
    """"
    upsates the frame
    """
    global fr, txt, boids
    if fr % 30 == 0: # 0.4, 1.2, 0.25, 0.8
        txt.text = " \n c = 0.4, a = 1.2, \n s = 0.25, w = 0.8 \n" + "N=5000; fps:" + f"{canvas.fps:0.1f}"
    fr += 1
    flocking(boids, perception, coeffs, asp, vrange, better_walls_w)
    propagate(boids, dt, vrange, arange)
    arrows.set_data(arrows=directions(boids, dt))

    canvas.render(alpha=False)
    if fr <= 2300:
        frame = canvas.render(alpha=False)
        writer.append_data(frame)
    else:
        writer.close()
        app.quit()


    #canvas.update()
    # w.append_data(i)
    # im


if __name__ == '__main__':
    timer = app.Timer(interval=0, start=True, connect=update)
    canvas.measure_fps()
    app.run()
