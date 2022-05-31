"""
Animations for various dynamical systems using `matplotlib`.

Code has been edited by Jeremiah Montemayor

Courtesy of Autonomous Systems Lab (ASL), Stanford
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.animation as animation

def animate_2D_quad(t, state, full_system = 1, frameDelay = 30,
                    mu = None, sigma = None):
    """Animate the planar quadrotor system from given position data.

       Input Arguments:
       state(n,8):  assumes the first four columns are x, z, θ, and ϕ
                    (i.e., x-position, z-position, pitch angle,
                    pendulum angle) in that order.

       full_system (boolean): true - quad+payload; false - quad only

       frameDelay (optional input): animation delay per frame in milliseconds

       Run the following lines to use this function:
       fig, ani = animate_2D_quad(t, state)
       ani.save("planar_quad_1.mp4", writer="ffmpeg")

    """
    # Extract the states
    x = state[:,0]
    z = state[:,1]
    theta = -state[:,2]

    if full_system:
      phi = -state[:,3]
    else:
      phi = np.zeros(x.shape)

    if mu is not None:
      mu = mu.copy()
      plotEst = bool(1)
      transpValue = 0.4
      mu[:,2] = -mu[:,2]
      mu[:,3] = -mu[:,3]
    else:
      plotEst = bool(0)

    # Geometry
    rod_width = 1.
    rod_height = 0.1
    axle_height = 0.2
    axle_width = 0.05
    prop_width = 0.5*rod_width
    prop_height = 1.5*rod_height
    hub_width = 0.3*rod_width
    hub_height = 2.5*rod_height
    pole_length = 1.
    pole_thick = 0.1
    ball_radius = 0.2

    # Figure and axis
    fig, ax = plt.subplots(dpi=100)
    x_min, x_max = np.min(x), np.max(x)
    x_pad = (rod_width + prop_width )/2 + 0.1*(x_max - x_min) + pole_length
    z_min, z_max = np.min(z), np.max(z)
    z_pad = (rod_width + prop_width)/2 + 0.1*(z_max - z_min) + pole_length
    ax.set_xlim([x_min - x_pad, x_max + x_pad])
    ax.set_ylim([z_min - z_pad, z_max + z_pad])
    ax.set_aspect('equal')

    # Quadrotor Artists
    rod = mpatches.Rectangle((-rod_width/2, -rod_height/2), rod_width, rod_height,
                             facecolor='tab:red', edgecolor='k')
    hub = mpatches.FancyBboxPatch((-hub_width/2, -hub_height/2),
                                  hub_width, hub_height,
                                  facecolor='tab:red', edgecolor='k',
                                  boxstyle='Round,pad=0.,rounding_size=0.05')
    axle_left = mpatches.Rectangle((-rod_width/2, rod_height/2),
                                   axle_width, axle_height,
                                   facecolor='tab:red', edgecolor='k')
    axle_right = mpatches.Rectangle((rod_width/2 - axle_width, rod_height/2),
                                    axle_width, axle_height,
                                    facecolor='tab:red', edgecolor='k')
    prop_left = mpatches.Ellipse(((axle_width - rod_width)/2,
                                  rod_height/2 + axle_height),
                                 prop_width, prop_height,
                                 facecolor='tab:gray', edgecolor='k',
                                 alpha=0.7)
    prop_right = mpatches.Ellipse(((rod_width - axle_width)/2,
                                   rod_height/2 + axle_height),
                                  prop_width, prop_height,
                                  facecolor='tab:gray', edgecolor='k',
                                  alpha=0.7)

    quadPatches = (rod, hub, axle_left, axle_right, prop_left, prop_right)

    # Pendulum Artists
    pole = mpatches.Arrow(0., 0., 0., -pole_length, width=pole_thick,
                              facecolor='tab:brown', edgecolor='k')
    ball = mpatches.Circle((0., -pole_length), radius=ball_radius,
                           facecolor='tab:orange', edgecolor='k')

    pendPatches = (pole, ball)

    # Add patches to axis
    for patch in quadPatches:
      if plotEst:
        patch.set_alpha(transpValue)
      ax.add_patch(patch)

    if full_system:
      for patch in pendPatches:
        if plotEst:
          patch.set_alpha(transpValue)
        ax.add_patch(patch)

    # Estimated Quadrotor and Pendulum Artists
    if plotEst:

      rod = mpatches.Rectangle((-rod_width/2, -rod_height/2), rod_width, rod_height,
                              facecolor='tab:red', edgecolor='k')
      hub = mpatches.FancyBboxPatch((-hub_width/2, -hub_height/2),
                                    hub_width, hub_height,
                                    facecolor='tab:red', edgecolor='k',
                                    boxstyle='Round,pad=0.,rounding_size=0.05')
      axle_left = mpatches.Rectangle((-rod_width/2, rod_height/2),
                                    axle_width, axle_height,
                                    facecolor='tab:red', edgecolor='k')
      axle_right = mpatches.Rectangle((rod_width/2 - axle_width, rod_height/2),
                                      axle_width, axle_height,
                                      facecolor='tab:red', edgecolor='k')
      prop_left = mpatches.Ellipse(((axle_width - rod_width)/2,
                                    rod_height/2 + axle_height),
                                  prop_width, prop_height,
                                  facecolor='tab:gray', edgecolor='k',
                                  alpha=0.7)
      prop_right = mpatches.Ellipse(((rod_width - axle_width)/2,
                                    rod_height/2 + axle_height),
                                    prop_width, prop_height,
                                    facecolor='tab:gray', edgecolor='k',
                                    alpha=0.7)

      estQuadPatches = (rod, hub, axle_left, axle_right, prop_left, prop_right)

      # Estimated Pendulum Artists
      pole = mpatches.Arrow(0., 0., 0., -pole_length, width=pole_thick,
                                facecolor='tab:brown', edgecolor='k')
      ball = mpatches.Circle((0., -pole_length), radius=ball_radius,
                            facecolor='tab:orange', edgecolor='k')

      estPendPatches = (pole, ball)

      for patch in estQuadPatches:
        ax.add_patch(patch)

      for patch in estPendPatches:
        ax.add_patch(patch)

      if sigma is not None:
        # Confidence ellipse
        eps = (1 - 0.95) / (2 * np.pi * np.sqrt(np.linalg.det(sigma[0,:2,:2])))
        C = -2 * np.log(eps) - 2 * np.log(2 * np.pi) - np.log(np.linalg.det(sigma[0,:2,:2]))
        errEllip = mpatches.Ellipse((0, 0),
                                width=2 * np.sqrt(C),
                                height=2 * np.sqrt(C),
                                facecolor="none", edgecolor = 'k')
        ax.add_patch(errEllip)

    # Trace and timestamp
    trace = ax.plot([], [], '--', linewidth=2, color='tab:blue')[0]
    timestamp = ax.text(0.1, 0.9, '', transform=ax.transAxes)

    # Animation
    def animate(k, t, x, z, theta, phi, mu = None, sigma = None):

        # Quad rotation and translation
        transformQuad = mtransforms.Affine2D().rotate_around(0., 0., theta[k])
        transformQuad += mtransforms.Affine2D().translate(x[k], z[k])
        transformQuad += ax.transData
        for patch in quadPatches:
            patch.set_transform(transformQuad)

        # Pole translation and rotation
        transformPend = mtransforms.Affine2D().rotate_around(0., 0., phi[k])
        transformPend += mtransforms.Affine2D().translate(x[k], z[k])
        transformPend += ax.transData
        for patch in pendPatches:
            patch.set_transform(transformPend)

        if plotEst:
            # Quad rotation and translation
            transformEstQuad = mtransforms.Affine2D().rotate_around(0., 0., mu[k,2])
            transformEstQuad += mtransforms.Affine2D().translate(mu[k,0], mu[k,1])
            transformEstQuad += ax.transData
            for patch in estQuadPatches:
                patch.set_transform(transformEstQuad)

            # Pole translation and rotation
            transformEstPend = mtransforms.Affine2D().rotate_around(0., 0., mu[k,3])
            transformEstPend += mtransforms.Affine2D().translate(mu[k,0], mu[k,1])
            transformEstPend += ax.transData
            for patch in estPendPatches:
              patch.set_transform(transformEstPend)

            if sigma is not None:
              eps = (1 - 0.95) / (2 * np.pi * np.sqrt(np.linalg.det(sigma[k,:2,:2])))
              C = -2 * np.log(eps) - 2 * np.log(2 * np.pi) - np.log(np.linalg.det(sigma[k,:2,:2]))
              errEllip.width = 2 * np.sqrt(C)
              errEllip.height = 2 * np.sqrt(C)

              T = np.eye(3)
              T[:2, :2] = np.linalg.cholesky(sigma[k,:2,:2])
              T[:2, 2] = mu[k,:2]
              transformEllip = mtransforms.Affine2D(T)
              transformEllip += ax.transData
              errEllip.set_transform(transformEllip)

        # Trace and time stamp animation
        trace.set_data(x[:k+1], z[:k+1])
        timestamp.set_text('t = {:.1f} s'.format(t[k]))

        # Consolidate artists
        artists = quadPatches + (trace, timestamp)
        if full_system:
            artists += pendPatches

        if plotEst:
            artists += estQuadPatches
            if full_system:
              artists += estPendPatches
            if sigma is not None:
              artists += (errEllip,)

        return artists

    if plotEst:
      ani = animation.FuncAnimation(fig, animate, t.size,
                                  fargs=(t, x, z, theta, phi,
                                         mu, sigma),
                                  interval=frameDelay, blit=True)
    else:
      ani = animation.FuncAnimation(fig, animate, t.size,
                                    fargs=(t, x, z, theta, phi),
                                    interval=frameDelay, blit=True)
    return fig, ani
