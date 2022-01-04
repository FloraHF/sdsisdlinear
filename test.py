
import numpy as np
from math import sqrt, sin, cos, tan, asin, atan2, pi
import matplotlib.pyplot as plt

from scipy.optimize import minimize

a = 1.2
r = .5
gmm = asin(1/a)

def ecurve_d(theta):
	xd = sin(theta) - a**2*sin(gmm)
	yd = -(cos(theta) - a**2*cos(gmm)) - a*(theta - gmm)

	xd = xd*r/(a**2 - 1)
	yd = yd*r/(a**2 - 1)

	return xd, yd

def ecurve_i(theta):
	xi = a**2*(sin(theta) - sin(gmm))
	yi = -a**2*(cos(theta) - cos(gmm)) - a*(theta - gmm)

	xi = xi*r/(a**2 - 1)
	yi = yi*r/(a**2 - 1)

	return xi, yi


def eslope_d(theta):
	kxd = cos(theta)
	kyd = sin(theta) - a
	norm = sqrt(kxd**2 + kyd**2)

	return kxd/norm, kyd/norm

def eslope_i(theta):
	kxi = a**2*cos(theta)
	kyi = a**2*sin(theta) - a
	norm = sqrt(kxi**2 + kyi**2)

	return kxi/norm, kyi/norm

def ebarrier_d(theta, d):
	x, y = ecurve_d(theta)
	kx, ky = eslope_d(theta)
	return x - d*kx, y - d*ky

def ebarrier_i(theta, d):
	x, y = ecurve_i(theta)
	kx, ky = eslope_i(theta)
	return x - d*kx, y - d*ky


def contact_theta(xd, yd):

	def slope_mismatch(theta, xd, yd):
		xd_, yd_ = ecurve_d(theta)
		kxd, kyd = eslope_d(theta)
		return ((xd - xd_)*kyd - (yd - yd_)*kxd)**2	

	theta = minimize(slope_mismatch, gmm/2, args=(xd, yd)).x.squeeze()

	# print('solution for contact point:', slope_mismatch(theta, xd, yd))

	return theta


def shift_range(xd, yd):

	def yd_mismatch(theta, yd):
		xd_, yd_ = ecurve_d(theta)
		return (yd_ - yd)**2

	# theta = minimize(yd_mismatch, gmm/2, args=(yd)).x.squeeze()
	theta = minimize(yd_mismatch, pi/2-gmm, args=(yd)).x.squeeze()
	xd_, yd_ = ecurve_d(theta)

	delta_min = xd - xd_
	delta_max = xd + yd/tan(pi/2 - gmm)

	# starting from (xd, yd), capture will happen at (delta, 0)

	# print('solution for delta_min', yd_mismatch(theta, yd))

	return delta_min, delta_max

def point_on_barrier_ddl(xd, yd, delta):
	
	# starting from (xd, yd), capture will happen at (delta, 0)
	# this is equivalent to start from (xd-delta, yd), and capture happens at (0, 0)
	xd_shift, yd_shift = xd-delta, yd

	# contact point on the defender's curved trajectory, in the shifted reference frame
	theta = contact_theta(xd_shift, yd_shift)
	xdc_shift, ydc_shift = ecurve_d(theta)

	# distance traveled by the defender
	d = sqrt((xdc_shift - xd_shift)**2 + (ydc_shift - yd_shift)**2)

	# position of the invader, in the shifted reference frame
	xi_shift, yi_shift = ebarrier_i(theta, d*a)

	return xi_shift + delta, yi_shift

def barrier_ddl(xd, yd):

	dmin, dmax = shift_range(xd, yd)

	xs, ys = [], []
	for dlt in np.linspace(dmin, dmax, 20):
		x, y = point_on_barrier_ddl(xd, yd, dlt)
		xs.append(x)
		ys.append(y)
	
	return np.asarray(xs), np.asarray(ys)


def barrier_ddlisaacs(y):
	
	xd, yd = 0, y
	dmin, dmax = shift_range(xd, yd)
	theta_min = contact_theta(xd-dmin, yd)
	theta_max = contact_theta(xd-dmax, yd)

	xs, ys = [], []
	for tht in np.linspace(theta_min, theta_max, 20):
		x, y = point_on_barrier_ddlisaacs(yd, tht)
		xs.append(x)
		ys.append(y)
	
	return np.asarray(xs), np.asarray(ys)

def point_on_barrier_ddlisaacs(y, theta):
	wk_ws = a*(gmm + sqrt(a**2 - 1) - theta)
	r_a2_1 = r/(a**2 - 1)

	tau = (y/r_a2_1 - wk_ws + cos(theta))/(a - sin(theta))
	yb = r_a2_1*(wk_ws - a**2*cos(theta) + a*(1-a*sin(theta))*tau)
	xb = r*(sin(theta) - tau*cos(theta))

	return xb, yb


def rotation_shift_range(xd, yd):

	def yd_mismatch(theta, yd):
		xd_, yd_ = ecurve_d(theta)
		return (yd_ - yd)**2

	theta = minimize(yd_mismatch, (gmm-pi/2)/2, args=(yd)).x.squeeze()
	xd_, yd_ = ecurve_d(theta)
	# starting from (xd_, yd_), capture will happen at (0, 0)
	# starting from (xd, yd), capture will happen at (delta, 0)
	# therefore, xd - delta = xd_

	shift = xd - xd_ # shift = delta
	max_rotate = pi - atan2(yd_, xd_) - (pi/2 - gmm)

	return shift, max_rotate

	# starting from (xd_, yd_), capture will happen at (0, 0)


def point_on_barrier_td(xd, yd, shift, rotate):
	
	# shift, max_rotate = rotation_shift_range(xd, yd)
	# shift such that capture happens at (0, 0)
	xd_shift, yd_shift = xd - shift, yd

	# rotate by rotate
	C = np.array([[cos(rotate), -sin(rotate)],
				  [sin(rotate),  cos(rotate)]])
	xy_rotate = C.dot(np.array([xd_shift, yd_shift])) # clockwise rotation
	xd_rotate, yd_rotate = xy_rotate[0], xy_rotate[1]

	# contact point on the defender's curved trajectory, in the shifted reference frame
	theta = contact_theta(xd_rotate, yd_rotate)
	xdc_rotate, ydc_rotate = ecurve_d(theta)

	# distance traveled by the defender
	d = sqrt((xdc_rotate - xd_rotate)**2 + (ydc_rotate - yd_rotate)**2)

	# position of the invader, in the shifted reference frame
	xi_rotate, yi_rotate = ebarrier_i(theta, d*a)

	# return the position in the original reference frame
	xi = C.transpose().dot(np.array([xi_rotate, yi_rotate]))

	return xi[0] + shift, xi[1]

def barrier_td(xd, yd):

	shift, max_rotate = rotation_shift_range(xd, yd)

	xs, ys = [], []
	for rot in np.linspace(0, max_rotate, 20):
		x, y = point_on_barrier_td(xd, yd, shift, rot)
		xs.append(x)
		ys.append(y)
	
	return np.asarray(xs), np.asarray(ys)


##################### plot ddl game barrier ##################
xd = 0
for yd in [1, 3.5, 5]:

	rx = [xd + r*cos(t) for t in np.linspace(-pi, pi, 50)]
	ry = [yd + r*sin(t) for t in np.linspace(-pi, pi, 50)]
	plt.plot(xd, yd, 'ro')
	plt.plot(rx, ry, 'r')

	xs, ys = barrier_td(xd, yd)
	plt.plot(xs, ys, 'b-')

	# xs, ys = barrier_ddlisaacs(yd)
	# plt.plot(xs, ys, 'bx')


plt.grid()
plt.axis('equal')
plt.show()
