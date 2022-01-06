
import numpy as np
from math import sqrt, sin, cos, tan, asin, atan2, pi
import matplotlib.pyplot as plt

from scipy.optimize import minimize

a = 1.2
r = .5
gmm = asin(1/a)

def frame_transform(x, y, ang=pi/2, xm=np.array([0, 0])):

	C = np.array([[ cos(ang), sin(ang)],
				  [-sin(ang), cos(ang)]])

	xy_ = C.dot(np.array([x, y]) - xm)

	return xy_[0], xy_[1]

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

def point_on_barrier_ddl(xd, yd, delta, transform={'ang':pi/2, 'xm':np.array([0, 0])}):
	
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

	xb, yb = xi_shift + delta, yi_shift
	if transform is not None:
		xb, yb = frame_transform(xb, yb, ang=transform['ang'], xm=transform['xm'])
		if xb < 0:
			xb = -xb
		if yb < 0:
			xb = -yb			

	return xb, yb

def barrier_ddl(xd, yd, transform={'ang':pi/2, 'xm':np.array([0, 0])}):

	dmin, dmax = shift_range(xd, yd)

	xs, ys = [], []
	for dlt in np.linspace(dmin, dmax, 20):
		x, y = point_on_barrier_ddl(xd, yd, dlt, transform=transform)
		xs.append(x)
		ys.append(y)
	
	return np.asarray(xs), np.asarray(ys)

def barrier_ddlisaacs(y, transform={'ang':pi/2, 'xm':np.array([0, 0])}):
	
	xd, yd = 0, y
	dmin, dmax = shift_range(xd, yd)
	theta_min = contact_theta(xd-dmin, yd)
	theta_max = contact_theta(xd-dmax, yd)

	xs, ys = [], []
	for tht in np.linspace(theta_min, theta_max, 20):
		x, y = point_on_barrier_ddlisaacs(yd, tht, transform=transform)
		xs.append(x)
		ys.append(y)
	
	return np.asarray(xs), np.asarray(ys)

def point_on_barrier_ddlisaacs(y, theta, transform={'ang':pi/2, 'xm':np.array([0, 0])}):
	wk_ws = a*(gmm + sqrt(a**2 - 1) - theta)
	r_a2_1 = r/(a**2 - 1)

	tau = (y/r_a2_1 - wk_ws + cos(theta))/(a - sin(theta))
	yb = r_a2_1*(wk_ws - a**2*cos(theta) + a*(1-a*sin(theta))*tau)
	xb = r*(sin(theta) - tau*cos(theta))

	if transform is not None:
		xb, yb = frame_transform(xb, yb, ang=transform['ang'], xm=transform['xm'])

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
	# max_rotate = pi/2 - atan2(yd_, xd_) - gmm

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
	xb, yb = xi[0] + shift, xi[1]

	return xb, yb

def barrier_td(xd, yd):

	shift, max_rotate = rotation_shift_range(xd, yd)

	# envelope barrier
	xs, ys = [], []
	for rot in np.linspace(0, max_rotate, 20):
		x, y = point_on_barrier_td(xd, yd, shift, rot)
		xs.append(x)
		ys.append(y)

	# natural barrier
	min_ddist = sqrt(yd**2 + (shift-xd)**2) - r
	min_idist = min_ddist*a
	for tht in np.linspace(max_rotate, 0, 20):
		x = shift - min_idist*cos(tht)
		y = 		min_idist*sin(tht)
		xs.append(x)
		ys.append(y)

	xs = [x - shift for x in xs]
	
	return np.asarray(xs), np.asarray(ys), shift


def capture_ring(xd, yd, transform={'ang':pi/2, 'xm':np.array([0, 0])}):

	rxs, rys = [], []
	for t in np.linspace(-pi, pi, 50):
		rx = xd + r*cos(t)
		ry = yd + r*sin(t)
		if transform is not None:
			rx, ry = frame_transform(rx, ry, ang=transform['ang'], xm=transform['xm'])
		rxs.append(rx)
		rys.append(ry)

	return np.asarray(rxs), np.asarray(rys)

##################### plot ddl game barrier ##################
def plot_barrier_ddlisaacs(transform={'ang':pi/2, 'xm':np.array([0, 0])}):

	'''DEC 4 Fig.11'''

	xd = 0
	for i, yd in enumerate([1, 2.99, 6]):

		dlabel = 'Defender' if i == 0 else None
		blabel = 'proposed barrier' if i == 0 else None
		blabel_isaacs = 'Isaacs barrier' if i == 0 else None

		# capture ring
		rx, ry = capture_ring(xd, yd, transform=transform)
		plt.plot(rx, ry, 'b')

		# defender locations
		if transform is not None:
			xd_, yd_ = frame_transform(xd, yd, ang=transform['ang'], xm=transform['xm'])
			plt.plot(xd_, yd_, 'bo', label=dlabel)
		else:
			plt.plot(xd, yd, 'bo', label=dlabel)
		
		# barrier computed by Isaacs
		xs, ys = barrier_ddlisaacs(yd, transform=transform)
		plt.plot(xs, ys, 'bx', label=blabel_isaacs)

		# barrier computed by the proposed method
		xs, ys = barrier_ddl(xd, yd, transform=transform)
		plt.plot(xs, ys, 'b-', label=blabel)

	plt.plot([0, 0], [-1, 5], 'r', lw=3, label='DDL')
	# plt.plot([-.5, 7], [0, 0], 'g', lw=3, label='Target')


	plt.grid()
	plt.legend(fontsize=12)
	plt.axis('equal')
	plt.xlabel('x', fontsize=12)
	plt.ylabel('y', fontsize=12)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)

	plt.show()

##################### plot td game barrier ##################
def plot_barrier_td(transform={'ang':pi/2, 'xm':np.array([0, 0])}):

	'''DEC 4 Fig.12'''

	xd = 0
	for i, yd in enumerate([1, 2.5, 4.5]):	

		dlabel = 'Defender' if i == 0 else None
		blabel = 'barrier' if i == 0 else None

		# shift, _ = rotation_shift_range(xd, yd)
		# transform['xm'] = np.array([shift, 0])
		# print(transform)

		# the target defense game barrier
		xs, ys, shift = barrier_td(xd, yd)
		if transform is not None:
			xs_, ys_ = [], []
			for x, y in zip(xs, ys):
				x_, y_ = frame_transform(x, y, ang=pi/2, xm=np.array([0, 0]))
				xs_.append(x_)
				ys_.append(y_)

			plt.plot(xs_, ys_, 'b-', label=blabel)
		else:
			plt.plot(xs, ys, 'b-', label=blabel)

		# the capture ring
		rx, ry = capture_ring(xd-shift, yd, transform=transform)
		plt.plot(rx, ry, 'b')

		# defender location
		if transform is not None:
			xd_, yd_ = frame_transform(xd-shift, yd, ang=transform['ang'], xm=transform['xm'])
			plt.plot(xd_, yd_, 'bo', label=dlabel)
		else:
			plt.plot(xd-shift, yd, 'bo', label=dlabel)		


	plt.plot([0, 0], [-.1, 6], 'r', lw=3, label='DDL')
	plt.plot([-.1, 6], [0, 0], 'g', lw=3, label='Target')

	plt.grid()
	plt.legend(fontsize=12)
	plt.axis('equal')
	plt.xlabel('x', fontsize=12)
	plt.ylabel('y', fontsize=12)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)

	plt.show()

##### call the functions
# plot_barrier_ddlisaacs()
# plot_barrier_td(transform=None)
plot_barrier_td()