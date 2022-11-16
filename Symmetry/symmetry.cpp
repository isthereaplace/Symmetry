#include <list>
#include <map>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <opencv2/cvconfig.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>

#ifdef HAVE_CUDA
#include <opencv2/cudaarithm.hpp>
#endif

using namespace std;
using namespace cv;

Mat radon(const Mat& img, const vector<Point2i>& bounds, int num_angles, bool reflect, vector<Mat>& rotated, vector<Rect>& boxes);

struct symmetryAxis {
	double rho;
	double theta;
	double phi;
	double val;
	symmetryAxis() : rho(0), theta(0), phi(-CV_PI), val(-1) {};
};

Rect2i extractBoundaries(const Mat& img, vector<Point2i>& bounds) {
	Mat inner;
	erode(img, inner, getStructuringElement(MORPH_CROSS, Size(3, 3)), Point(-1,-1), 1, BORDER_CONSTANT, 0);
		
	bitwise_xor(img, inner, inner);
	Rect2i bbox(img.cols - 1, img.rows - 1, 0, 0);

	vector<vector<Point2i>> contours;
	findContours(inner, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	for (int i = 1; i < contours.size(); i++) {
		std::move(contours[i].begin(), contours[i].end(), std::back_inserter(contours[0]));
	}
	convexHull(contours[0], bounds);

	for (auto b : bounds) {
		bbox.x = min(bbox.x, b.x);
		bbox.y = min(bbox.y, b.y);
		bbox.width = max(bbox.width, b.x);
		bbox.height = max(bbox.height, b.y);
	}
	bbox.width -= (bbox.x - 1);
	bbox.height -= (bbox.y - 1);
	return bbox;
}

int reflectionalMeasure(const Mat& img, Vec2d c, Vec3d l, bool show) {
	double d = sqrt(img.rows * img.rows + img.cols * img.cols);
	double x0 = l[0] * cos(l[1]);
	double y0 = l[0] * sin(l[1]);
	double dot = (c[0] - x0) * sin(l[1]) - (c[1] - y0) * cos(l[1]);
	double xc = x0 + dot * sin(l[1]);
	double yc = y0 - dot * cos(l[1]);
	double beta = l[2] > -CV_PI ? l[2] : 0;
	double k = tan(beta);

	Matx33d M0(1, 0, -xc, 0, 1, -yc, 0, 0, 1);
	Matx33d M1(cos(l[1]), sin(l[1]), 0, -sin(l[1]) - k * cos(l[1]), cos(l[1]) - k * sin(l[1]), 0, 0, 0, 1);
	Matx33d M2(-1, 0, 0, 0, 1, 0, 0, 0, 1);
	Matx33d M3 = M1.inv();
	Matx33d M4(1, 0, xc, 0, 1, yc, 0, 0, 1);
	Matx33d M5 = M4 * M3 * M2 * M1 * M0;
	Matx23d M(M5(0, 0), M5(0, 1), M5(0, 2), M5(1, 0), M5(1, 1), M5(1, 2));
	Mat ref;
	warpAffine(img, ref, M, Size(img.cols, img.rows));
	Mat both;
	bitwise_and(img, ref, both);

	if (show) {
		Mat canvas;
		Mat green;
		bitwise_and(1 - img, 1 - ref, green);
		merge(vector<Mat>({ 1 - img, green, 1 - ref}), canvas);
		line(canvas, Point2d(xc - d * sin(l[1]), yc + d * cos(l[1])), Point2d(xc + d * sin(l[1]), yc - d * cos(l[1])),
			 Scalar(0, 255, 0), 1, LINE_AA);
		if (l[2] > -CV_PI) {
			double ang = l[1] - CV_PI/2 + beta;
			line(canvas, Point2d(xc - d * sin(ang), yc + d * cos(ang)), Point2d(xc + d * sin(ang), yc - d * cos(ang)),
				 Scalar(255, 0, 255), 1, LINE_AA);
		}
		imshow("Intersection Area", 255 * canvas);
		int k = waitKey(0);
	}
	return countNonZero(both);
}

void rotateImage(const Mat& img, Mat& rot, double ang, Vec2d xrange, const vector<Point2i>& bounds) {
	double ymin = std::numeric_limits<int>::max();
	double ymax = std::numeric_limits<int>::min();
	double cosx = cos(ang);
	double sinx = sin(ang);
	for (auto b : bounds) {
		double y = -b.x * sinx + b.y * cosx;
		ymin = min<int>(ymin, floor(y));
		ymax = max<int>(ymax, ceil(y));
	}
	Matx23d M(cos(ang), sin(ang), -xrange[0], -sin(ang), cos(ang), -ymin);
	warpAffine(img, rot, M, Size(round(xrange[1] - xrange[0]) + 1, ymax - ymin + 1));
}

void collectStripes(const Mat& img, vector<list<Vec2i>>& stripes) {
	stripes.resize(img.rows);
	Vec2i str;
	uchar* ptr = img.data;
	for (int y = 0; y < img.rows; y++) {
		bool prev = false;
		for (int x = 0; x > -img.cols; x--) {
			bool curr = *ptr++;
			if (curr && !prev) {
				str[1] = x;
			}
			if (!curr && prev) {
				str[0] = x;
				stripes[y].push_back(str);
			}
			prev = curr;
		}
		if (prev) {
			str[0] = -img.cols;
			stripes[y].push_back(str);
		}
	}
}

int getMinIndex(int idx_sum, int cols) {
	if (idx_sum <= cols - 1) {
		return 0;
	}
	else {
		return idx_sum - cols + 1;
	}
}

pair<double, double> alignStripes(const vector<list<Vec2i>>& stripes, int idx_sum) {
	int jmin = getMinIndex(idx_sum, stripes.size());
	int add = 0;
	if (idx_sum % 2 == 0) {
		for (auto a : stripes[idx_sum / 2]) {
			add += a[1] - a[0];
		}
	}
	int num_pairs = 0;
	for (int j = jmin; j <= (idx_sum - 1) / 2; j++) {
		num_pairs += stripes[j].size() * stripes[idx_sum - j].size();
	}
	vector<pair<double, int>> changes(4 * num_pairs);
	pair<double, int>* ptr = changes.data();

	for (int j = jmin; j <= (idx_sum - 1) / 2; j++) {
		int k = idx_sum - j;
		for (auto a : stripes[j]) {
			for (auto b : stripes[k]) {
				double dx = k - j;
				double len = min(a[1] - a[0], b[1] - b[0]);
				double t1 = (b[0] - a[1]) / dx;
				double t2 = min(b[0] - a[0], b[1] - a[1]) / dx;
				double t3 = max(b[0] - a[0], b[1] - a[1]) / dx;
				double t4 = (b[1] - a[0]) / dx;
				int dy = 2 * (k - j);
				*ptr++ = { t1, +dy };
				*ptr++ = { t2, -dy };
				*ptr++ = { t3, -dy};
				*ptr++ = { t4, +dy };
			}
		}
	}
	double angle = 0;
	int best_val = 0;
	if (!changes.empty()) {
		sort(changes.begin(), changes.end(), [](pair<double, int> a, pair<double, int> b) { return a.first < b.first; });
		vector<pair<double, double>> values(changes.size());
		values[0] = { -std::numeric_limits<double>::max(), add };
		// Collecting derivatives
		for (int i = 1; i < changes.size(); i++) {
			changes[i].second += changes[i - 1].second;
			values[i] = { changes[i].first, changes[i - 1].second * (changes[i].first - changes[i - 1].first) };
			values[i].second += values[i - 1].second;
			if (values[i].second > best_val) {
				best_val = values[i].second;
				angle = atan2(values[i].first, 1);
			}
		}
	}
	return { angle, best_val };
}

void getDistRanges(int rows, int cols, double ang, double& xmin, double& xmax) {
	xmin = 0;
	xmax = 0;
	xmin = min(xmin, 0 * cos(ang) + (rows - 1) * sin(ang));
	xmin = min(xmin, (cols - 1) * cos(ang) + 0 * sin(ang));
	xmin = min(xmin, (cols - 1) * cos(ang) + (rows - 1) * sin(ang));
	xmax = max(xmax, 0 * cos(ang) + (rows - 1) * sin(ang));
	xmax = max(xmax, (cols - 1) * cos(ang) + 0 * sin(ang));
	xmax = max(xmax, (cols - 1) * cos(ang) + (rows - 1) * sin(ang));
}

symmetryAxis firstApproach(Mat& img, int area, Point2d& center, bool allow_shear, const vector<Point2i>& bounds) {
	// Initial symmetry axis as one of the principal axes
	double d = sqrt(img.rows * img.rows + img.cols * img.cols);
	Moments data = moments(img);
	center.x = data.m10 / data.m00;
	center.y = data.m01 / data.m00;
	double angle = atan2(2 * data.mu11, data.mu20 - data.mu02) / 2 + CV_PI/2;
	
	symmetryAxis best;
	if (allow_shear) {
		best.val = std::numeric_limits<double>::max();
		int nang = 180;
		int ndiv = 90;
		for (int i = 0; i < nang; i++) {
			double alpha = i * (CV_PI / nang);
			double a11 = cos(alpha);
			double a12 = sin(alpha);
			double p00 = data.m00;
			double p10 = data.m10 * a11 + data.m01 * a12;
			double p20 = data.m20 * pow(a11, 2) + 2 * data.m11 * a11 * a12 + data.m02 * pow(a12, 2);
			double p30 = data.m30 * pow(a11, 3) + 3 * data.m21 * pow(a11, 2) * a12 + 3 * data.m12 * a11 * pow(a12, 2) + data.m03 * pow(a12, 3);
			
			double q20 = p20 - pow(p10, 2) / p00;
			double q30 = p30 - 3 * p20 * p10 / p00 + 2 * pow(p10, 3) / pow(p00, 2);

			for (int j = -ndiv + 1; j <= ndiv - 1; j++) {
				double beta = j * CV_PI/2 / ndiv;
				double k = tan(beta);
				double a21 = -sin(alpha) - k * cos(alpha);
				double a22 = +cos(alpha) - k * sin(alpha);

				double p01 = data.m10 * a21 + data.m01 * a22;
				double p11 = data.m20 * a11 * a21 + data.m11 * (a11 * a22 + a21 * a12) + data.m02 * a12 * a22;
				double p02 = data.m20 * pow(a21, 2) + 2 * data.m11 * a21 * a22 + data.m02 * pow(a22, 2);
				double p21 = data.m30 * pow(a11, 2) * a21 + data.m21 * pow(a11, 2) * a22 + 2 * data.m21 * a11 * a12 * a21 + 2 * data.m12 * a11 * a12 * a22 + data.m12 * pow(a12, 2) * a21 + data.m03 * pow(a12, 2) * a22;
				double p12 = data.m30 * pow(a21, 2) * a11 + data.m21 * pow(a21, 2) * a12 + 2 * data.m21 * a21 * a22 * a11 + 2 * data.m12 * a21 * a22 * a12 + data.m12 * pow(a22, 2) * a11 + data.m03 * pow(a22, 2) * a12;
				double p03 = data.m30 * pow(a21, 3) + 3 * data.m21 * pow(a21, 2) * a22 + 3 * data.m12 * a21 * pow(a22, 2) + data.m03 * pow(a22, 3);
				
				double q11 = p11 - p10 * p01 / p00;
				double q02 = p02 - pow(p01, 2) / p00;				
				double q21 = p21 - p20 * p01 / p00 - 2 * p11 * p10 / p00 + 2 * pow(p10, 2) * p01 / pow(p00, 2);
				double q12 = p12 - p02 * p10 / p00 - 2 * p11 * p01 / p00 + 2 * pow(p01, 2) * p10 / pow(p00, 2);
				double q03 = p03 - 3 * p02 * p01 / p00 + 2 * pow(p01, 3) / pow(p00, 2);

				double fval = abs(q30) + abs(q03) + pow(abs(q11), 1.5);	// Moment-based functional that is close to zero for symmetric figures

				if (fval < best.val) {
					best.val = fval;
					best.theta = alpha;
					best.rho = center.x * cos(best.theta) + center.y * sin(best.theta);
					best.phi = beta;
				}
			}
		}
		best.val = reflectionalMeasure(img, center, Vec3d(best.rho, best.theta, best.phi), false) / double(area);
		double theta = best.theta - CV_PI/2 + best.phi;
		double phi = -best.phi;
		double rho = center.x * cos(theta) + center.y * sin(theta);
		double val = reflectionalMeasure(img, center, Vec3d(rho, theta, phi), false) / double(area);

		if (val > best.val) {
			best.val = val;
			best.rho = rho;
			best.theta = theta;
			best.phi = phi;
		}
		return best;
	}
	
	for (double theta : {angle, angle - CV_PI/2}) {
		double rho = center.x * cos(theta) + center.y * sin(theta);
		if (!allow_shear) {
			double val = reflectionalMeasure(img, center, Vec3d(rho, theta, -CV_PI), false) / double(area);
			if (val > best.val) {
				best.val = val;
				best.rho = rho;
				best.theta = theta;
			}
		}
		else {
			double xmin, xmax;
			getDistRanges(img.rows, img.cols, theta, xmin, xmax);
			double dx = rho - floor(rho);
			xmin = floor(xmin - dx) + dx;
			xmax = ceil(xmax - dx) + 1 + dx;
			Mat rot;
			rotateImage(img, rot, theta, Vec2d(xmin, xmax), bounds);
			int cur_area = countNonZero(rot);
			int idx = round(rho - xmin);
			vector<list<Vec2i>> stripes;
			collectStripes(rot, stripes);
			auto res = alignStripes(stripes, 2 * idx);
			if (res.second / double(cur_area) > best.val) {
				best.val = res.second / double(cur_area);
				best.rho = rho;
				best.theta = theta;
				best.phi = res.first;
			}
		}
	}
	return best;
}

struct procParams{
	int num_angles;
	bool allow_shear;
	bool use_cuda;
	bool use_matmul;
	bool rotational;
	int degree;
	int num_subdiv;
	bool visualize;
	bool invert;
	int num_checks;
	procParams(const int argc, const char** argv) {
		num_angles = 180;
		allow_shear = false;
		use_cuda = false;
		use_matmul = false;
		rotational = false;
		degree = 0;
		num_subdiv = 1;
		visualize = false;
		invert = false;
		num_checks = 0;
		for (int i = 2; i < argc; i += 2) {
			if (!strcmp(argv[i], "-a")) {
				num_angles = atoi(argv[i + 1]);
			}
			if (!strcmp(argv[i], "-s")) {
				allow_shear = atoi(argv[i + 1]);
			}
			if (!strcmp(argv[i], "-c")) {
				use_cuda = atoi(argv[i + 1]);
			}
			if (!strcmp(argv[i], "-m")) {
				use_matmul = atoi(argv[i + 1]);
			}
			if (!strcmp(argv[i], "-t")) {
				rotational = atoi(argv[i + 1]);
			}
			if (!strcmp(argv[i], "-d")) {
				degree = atoi(argv[i + 1]);
			}
			if (!strcmp(argv[i], "-k")) {
				num_subdiv = atoi(argv[i + 1]);
			}
			if (!strcmp(argv[i], "-v")) {
				visualize = atoi(argv[i + 1]);
			}
			if (!strcmp(argv[i], "-i")) {
				invert = atoi(argv[i + 1]);
			}
			if (!strcmp(argv[i], "-n")) {
				num_checks = atoi(argv[i + 1]);
			}
		}
	}
};


double binom(int n, int k) {
	if (k > n) {
		return 0;
	}
	return std::tgamma(n + 1) / (std::tgamma(n - k + 1) * std::tgamma(k + 1));
}


double rotationalMeasure(const vector<Mat>& images, const vector<Point2d>& shifts, int area, Point2d center) {
	int degree = images.size();
	int val = 0;
	for (int i = 1; i <= degree / 2; i++) {
		double alpha = i * CV_2PI / degree;
		int dx = round(+center.x * (cos(alpha) - 1) + center.y * (sin(alpha) - 0) - shifts[0].x + shifts[i].x);
		int dy = round(-center.x * (sin(alpha) - 0) + center.y * (cos(alpha) - 1) - shifts[0].y + shifts[i].y);

		int x1 = dx >= 0 ? 0 : -dx;
		int x2 = dx >= 0 ? +dx : 0;
		int w = min<int>(images[0].cols - x1, images[i].cols - x2);

		int y1 = dy >= 0 ? 0 : -dy;
		int y2 = dy >= 0 ? +dy : 0;
		int h = min<int>(images[0].rows - y1, images[i].rows - y2);

		Rect roi1(x1, y1, w, h);
		Rect roi2(x2, y2, w, h);
		Mat both;
		bitwise_and(images[0](roi1), images[i](roi2), both);
		int coef = 2 * i == degree ? degree / 2 : degree;
		val += coef * countNonZero(both);
	}
	return double(val) / (area * degree * (degree - 1) / 2);
}

vector<double> overlayColumns(const Mat& accum, int area, int irow, int jrow) {
	vector<double> dx(2 * accum.cols - 1, 0);
	int num_angles = accum.rows;
	int imin = accum.cols;
	int imax = -1;
	int jmin = accum.cols;
	int jmax = -1;

	for (int t = 0; t <= accum.cols - 1 && imin == accum.cols; t++) {
		if (accum.at<int32_t>(irow, t)) {
			imin = t;
		}
	}
	for (int t = accum.cols - 1; t >= 0 && imax == -1; t--) {
		if (accum.at<int32_t>(irow, t)) {
			imax = t;
		}
	}
	for (int t = 0; t <= accum.cols - 1 && jmin == accum.cols; t++) {
		if (accum.at<int32_t>(jrow, t)) {
			jmin = t;
		}
	}
	for (int t = accum.cols - 1; t >= 0 && jmax == -1; t--) {
		if (accum.at<int32_t>(jrow, t)) {
			jmax = t;
		}
	}

	for (int d = jmin - imax; d <= jmax - imin; d++) {
		int tmin = d <= 0 ? 0 : d;
		int tmax = d >= 0 ? accum.cols - 1 : accum.cols - 1 + d;
		int idx = d + accum.cols - 1;
		if (irow <= num_angles && jrow <= num_angles) {
			for (int t = tmin; t <= tmax; t++) {
				dx[idx] += min<int32_t>(accum.at<int32_t>(irow, t - d), accum.at<int32_t>(jrow, t));
			}
		}
	}
	for (int t = 0; t < dx.size(); t++) {
		dx[t] /= area;
	}
	return dx;
}

int gcd(int a, int b) {
	if (b == 0)
		return a;
	return gcd(b, a % b);
}

int lcm(int a, int b) {
	return a * b / gcd(a, b);
}

Mat radonOld(const Mat& img, int num_angles, bool reflect) {
	UMat _img;
	img.copyTo(_img);
	UMat _accum;
	HoughAccum(_img, _accum, 1, CV_PI / num_angles, 0);
	if (reflect) {
		Mat accum(2 * (_accum.rows - 2), _accum.cols - 2, _accum.type());
		_accum(Rect(1, 1, accum.cols, accum.rows / 2)).copyTo(accum(Rect(0, 0, accum.cols, accum.rows / 2)));
		flip(_accum, _accum, 1);
		_accum(Rect(1, 1, accum.cols, accum.rows / 2)).copyTo(accum(Rect(0, accum.rows / 2, accum.cols, accum.rows / 2)));
		return accum;
	}
	Mat accum(_accum.rows - 2, _accum.cols - 2, _accum.type());
	_accum(Rect(1, 1, accum.cols, accum.rows)).copyTo(accum(Rect(0, 0, accum.cols, accum.rows)));
	return accum;
}

int getDegree(const Mat& img, int area, const vector<Point2i>& bounds) {

	int num_degrees = 180;
	vector<Mat> rotated;
	vector<Rect> boxes;
	Mat accum = radon(img, bounds, num_degrees, false, rotated, boxes);

	int num_parts = 5;
	Mat sections(num_degrees, 2*num_parts - 1, CV_32F);
	int32_t* ptr0 = (int32_t*)accum.data;
	for (int i = 0; i < accum.rows; i++) {
		double area = sum(accum.row(i))(0);
		int prev = 0;
		int k = 0;
		double frac = area * (k + 1) / (2 * num_parts);
		for (int j = 0; j < accum.cols; j++) {
			int curr = prev + *ptr0++;
			while (frac >= prev && frac < curr) {
				sections.at<float>(i, k++) = j - 1 + (frac - prev) / (curr - prev);
				frac = area * (k + 1) / (2 * num_parts);
			}
			prev = curr;
		}
	}
	Mat left = Mat::zeros(sections.rows, num_parts - 1, CV_32F);
	Mat right = Mat::zeros(sections.rows, num_parts - 1, CV_32F);
	for (int i = 0; i < num_parts - 1; i++) {
		left.col(i) = sections.col(num_parts - 1) - sections.col(num_parts - 2 - i);
		right.col(i) = sections.col(num_parts + i) - sections.col(num_parts - 1);
	}
	vconcat(left, right, left);
	vconcat(right, left(Rect(0, 0, right.cols, right.rows)), right);

	Mat data;
	merge(vector<Mat>({ left, right }), data);
	Mat res;
	cv::dft(data.t(), res, DFT_ROWS | DFT_COMPLEX_INPUT | DFT_COMPLEX_OUTPUT);
	vector<Mat> comps;
	split(res, comps);
	cv::sqrt(comps[0].mul(comps[0]) + comps[1].mul(comps[1]), res);

	vector<int> best_idx(num_parts - 1, 0);
	vector<float> best_val(num_parts - 1, -1);
	float* ptr = (float*)res.data;
	for (int i = 0; i < res.rows; i++) {
		for (int j = 0; j < res.cols; j++) {
			if (*ptr > best_val[i] && j >= 2) {
				best_val[i] = *ptr;
				best_idx[i] = j;
			}
			ptr++;
		}
	}

	int deg = -1;
	int freq = 0;
	for (int i = 0; i < best_idx.size(); i++) {
		int temp = 1;
		for (int j = i + 1; best_idx[i] >= 0 && j < best_idx.size(); j++) {
			if (best_idx[j] == best_idx[i]) {
				best_idx[j] = -1;
				temp++;
			}
		}
		if (temp > freq || temp == freq && best_idx[i] > deg) {
			deg = best_idx[i];
			freq = temp;
		}
	}

	return deg;
}

Mat radon(const Mat& img, const vector<Point2i>& bounds, int num_angles, bool reflect, vector<Mat>& rotated, vector<Rect>& boxes) {
	int diag = ceil(sqrt(img.rows * img.rows + img.cols * img.cols));
	Mat accum = Mat::zeros(num_angles, 2 * diag + 1, CV_32S);

	int k = (num_angles % 2 == 0) ? num_angles / 2 : num_angles;
	rotated.resize(num_angles);
	boxes.resize(num_angles);

	for (int i = 0; i < k; i++) {
		double ang = CV_PI * i / num_angles;
		double cosx = cos(ang);
		double sinx = sin(ang);
		int xmin = std::numeric_limits<int>::max();
		int xmax = std::numeric_limits<int>::min();
		int ymin = std::numeric_limits<int>::max();
		int ymax = std::numeric_limits<int>::min();
		for (auto b : bounds) {
			double xt = +b.x * cosx + b.y * sinx;
			double yt = -b.x * sinx + b.y * cosx;
			xmin = min<int>(xmin, floor(xt));
			xmax = max<int>(xmax, ceil(xt));
			ymin = min<int>(ymin, floor(yt));
			ymax = max<int>(ymax, ceil(yt));
		}
		Matx23d M(cosx, sinx, -xmin, -sinx, cosx, -ymin);
		warpAffine(img, rotated[i], M, Size(xmax - xmin + 1, ymax - ymin + 1));
		Mat sumd;
		reduce(rotated[i], sumd, 0, REDUCE_SUM, CV_32S);
		sumd.copyTo(accum(Rect(xmin + diag, i, xmax - xmin + 1, 1)));
		boxes[i] = Rect(Point2i(xmin, ymin), Point2i(xmax, ymax));
		if (num_angles % 2 == 0) {
			reduce(rotated[i], sumd, 1, REDUCE_SUM, CV_32S);
			sumd = sumd.t();
			sumd.copyTo(accum(Rect(ymin + diag, i + num_angles / 2, ymax - ymin + 1, 1)));
			// rotate(rotated[i], rotated[i + num_angles / 2], ROTATE_90_COUNTERCLOCKWISE);
			boxes[i + num_angles/2] = Rect(Point2i(ymin, -xmax), Point2i(ymax, -xmin));
		}
	}

	if (reflect) {
		Mat temp;
		flip(accum, temp, 1);
		vconcat(accum, temp, accum);
	}
	return accum;
}

void third_check_old(vector<Mat>& rotated, int i, procParams params, list<std::pair<Vec2i, double>>& new_cands, vector<int>& rot_area, symmetryAxis best) {
	
	if (new_cands.size() * rotated[i].rows * rotated[i].cols < params.num_checks * rot_area[i]) {
		return;
	}
	
	Mat horsum;
	if (rotated[i].empty()) {
		rotate(rotated[i - params.num_angles / 2], rotated[i], ROTATE_90_COUNTERCLOCKWISE);
	}
	rotated[i].convertTo(horsum, CV_32S);
	for (int y = 0; y < horsum.rows; y++) {
		int32_t* ptr0 = (int32_t*)(horsum.data + y * horsum.step);
		int32_t* ptr1 = ptr0 + 1;
		for (int j = 1; j < horsum.cols; j++) {
			if (*ptr1) {
				*ptr1++ = ++(*ptr0++);
			}
			else {
				*ptr1++ = *ptr0++;
			}
		}
	}
	auto cand = new_cands.begin();
	while (cand != new_cands.end()) {
		int j = cand->first[1];
		Mat left = 2 * horsum.col(j / 2);
		if (j % 2 == 0) {
			left -= rotated[i].col(j / 2);
		}
		double val = sum(min(left, 2 * horsum.col(horsum.cols - 1) - left))(0) / double(rot_area[i]);
		if (val > best.val) {
			cand->second = min(cand->second, val);
			cand++;
		}
		else {
			cand = new_cands.erase(cand);
		}
	}
}

void third_filter(double angle, list<pair<Vec2i, double>>& new_cands, vector<int>& dmin, vector<int>& dmax, Mat& accum,
	vector<int>& rot_area, double best_val)
{
	if (new_cands.empty()) {
		return;
	}
	int i = new_cands.front().first[0];
	int num_angles = accum.rows;
	int davg = (accum.cols - 1) / 2;

	int ang_steps = round(angle * num_angles / CV_PI);
	int left = (i - ang_steps);
	int right = (i + ang_steps);

	bool invert = false;
	if (left < 0) {
		int k = left / num_angles - 1;
		left -= k * num_angles;
		invert = (abs(k) % 2) == 1;
	}
	if (right >= num_angles) {
		int k = right / num_angles;
		right -= k * num_angles;
		invert = (k % 2) == 1 ? !invert : invert;
	}
	double ang_left = CV_PI * left / num_angles;
	double ang_right = CV_PI * right / num_angles;
	double ang_center = CV_PI * i / num_angles;

	int idx1 = std::numeric_limits<int>::max();
	int idx2 = std::numeric_limits<int>::min();
	for (auto cand : { new_cands.front(), new_cands.back() }) {
		double rad = cand.first[1] / 2.0 + (dmin[i] - davg);
		double xc = rad * cos(ang_center);
		double yc = rad * sin(ang_center);
		double meet = xc * cos(ang_left) + yc * sin(ang_left);
		if (invert == false) {
			int jmin = max<int>(dmin[left], floor(-dmax[right] + 2 * meet + 2 * davg));
			int j1 = ceil(-jmin + 2 * meet + 2 * davg);
			idx1 = min(idx1, jmin + j1 - 1);
			idx2 = max(idx2, jmin + j1 - 0);
		}
		else {
			int jmin = max<int>(dmin[left], floor(dmin[right] + 2 * meet));
			int j1 = floor(jmin - 2 * meet);
			idx1 = min(idx1, jmin - j1 - 1);
			idx2 = max(idx2, jmin - j1 - 0);
		}
	}

	vector<int> sums(idx2 - idx1 + 1, -1);
	auto cand = new_cands.begin();
	while (cand != new_cands.end()) {

		double rad = cand->first[1] / 2.0 + (dmin[i] - davg);
		double xc = rad * cos(ang_center);
		double yc = rad * sin(ang_center);
		double meet = xc * cos(ang_left) + yc * sin(ang_left);

		int sum1 = 0;
		int sum2 = 0;
		double alpha;
		if (invert == false) {
			int jmin = max<int>(dmin[left], floor(-dmax[right] + 2 * meet + 2 * davg));
			int jmax = min<int>(dmax[left], ceil(-dmin[right] + 2 * meet + 2 * davg));
			alpha = -jmin + 2 * meet + 2 * davg;
			int j1 = ceil(alpha);
			alpha = j1 - alpha;

			int32_t* ptrl = (int32_t*)accum.data + left * accum.step1() + jmin;
			int32_t* ptrr = (int32_t*)accum.data + right * accum.step1() + j1;
			sum1 = sums[jmin + j1 - 0 - idx1];
			sum2 = sums[jmin + j1 - 1 - idx1];
			if (sum1 < 0 && sum2 < 0) {
				sum1 = sum2 = 0;
				for (int j = jmin; j <= jmax; j++) {
					sum1 += min<int>(*ptrl, *ptrr--);
					sum2 += min<int>(*ptrl++, *ptrr);
				}
				sums[jmin + j1 - 0 - idx1] = sum1;
				sums[jmin + j1 - 1 - idx1] = sum2;
			}
			else if (sum1 < 0 && sum2 >= 0) {
				sum1 = 0;
				for (int j = jmin; j <= jmax; j++) {
					sum1 += min<int>(*ptrl++, *ptrr--);
				}
				sums[jmin + j1 - 0 - idx1] = sum1;
			}
			else if (sum1 >= 0 && sum2 < 0) {
				sum2 = 0;
				for (int j = jmin; j <= jmax; j++) {
					sum2 += min<int>(*ptrl++, *--ptrr);
				}
				sums[jmin + j1 - 1 - idx1] = sum2;
			}
		}
		else {
			int jmin = max<int>(dmin[left], floor(dmin[right] + 2 * meet));
			int jmax = min<int>(dmax[left], ceil(dmax[right] + 2 * meet));
			alpha = jmin - 2 * meet;
			int j1 = floor(alpha);
			alpha = alpha - j1;

			int32_t* ptrl = (int32_t*)accum.data + left * accum.step1() + jmin;
			int32_t* ptrr = (int32_t*)accum.data + right * accum.step1() + j1;
			sum1 = sums[jmin - j1 - 0 - idx1];
			sum2 = sums[jmin - j1 - 1 - idx1];
			if (sum1 < 0 && sum2 < 0) {
				sum1 = sum2 = 0;
				for (int j = jmin; j <= jmax; j++) {
					sum1 += min<int>(*ptrl, *ptrr++);
					sum2 += min<int>(*ptrl++, *ptrr);
				}
				sums[jmin - j1 - 0 - idx1] = sum1;
				sums[jmin - j1 - 1 - idx1] = sum2;
			}
			else if (sum1 < 0 && sum2 >= 0) {
				sum1 = 0;
				for (int j = jmin; j <= jmax; j++) {
					sum1 += min<int>(*ptrl++, *ptrr++);
				}
				sums[jmin - j1 - 0 - idx1] = sum1;
			}
			else if (sum1 >= 0 && sum2 < 0) {
				sum2 = 0;
				for (int j = jmin; j <= jmax; j++) {
					sum2 += min<int>(*ptrl++, *++ptrr);
				}
				sums[jmin - j1 - 1 - idx1] = sum2;
			}
		}
		double rate = ((1 - alpha)*sum1 + alpha * sum2) / rot_area[i];
		if (rate > best_val) {
			cand->second = min(cand->second, rate);
			cand++;
		}
		else {
			cand = new_cands.erase(cand);
		}
	}
}

int main(const int argc, const char** argv) {
	if (argc < 2) {
		printf("Wrong number of arguments\n");
		return -1;
	}
	procParams params(argc, argv);

	#ifdef HAVE_CUDA
	// Senseless operation just to load and check CUDA
	if (params.use_cuda) {
		Mat temp = Mat::zeros(Size(1, 1), CV_32F);
		Mat res;
		cv::cuda::gemm(temp.t(), temp, 1, Mat(), 0, res);
	}
	#endif

	auto start = chrono::high_resolution_clock::now();

	const char* image_path = argv[1];
	Mat img = imread(image_path, IMREAD_GRAYSCALE);
	threshold(img, img, 127, 1, params.invert ? THRESH_BINARY : THRESH_BINARY_INV);
	
	vector<Point2i> bounds;
	Rect2i bbox = extractBoundaries(img, bounds);
	img = img(bbox);
	int area = countNonZero(img);
	for (auto& b : bounds) {
		b.x -= bbox.x;
		b.y -= bbox.y;
	}

	if (params.rotational) {
		if (params.degree == 0) {
			params.degree = getDegree(img, area, bounds);
		}
		if (params.degree % 2 == 1 && params.num_subdiv % 2 == 0) {
			params.num_subdiv *= 2;
		}
		params.num_angles = params.degree * params.num_subdiv;
		if (params.num_angles % 2 == 0) {
			params.num_angles /= 2;
		}
		vector<Mat> rotated;
		vector<Rect> boxes;
		Mat accum = radon(img, bounds, params.num_angles, true, rotated, boxes);
		if (accum.rows % 4 == 0) {
			for (int i = 0; i < accum.rows / 4; i++) {
				if (rotated[i + accum.rows / 4].empty()) {
					rotate(rotated[i], rotated[i + accum.rows / 4], ROTATE_90_COUNTERCLOCKWISE);
				}
			}
		}

		vector<Mat> images(params.degree);
		vector<Point2d> shifts(params.degree);
		for (int i = 0; i < params.degree; i++) {
			int idx = (i * accum.rows / params.degree) % (accum.rows / 2);
			if (i <= (params.degree - 1) / 2) {
				images[i] = rotated[idx];
				shifts[i] = -boxes[idx].tl();
			}
			else {
				rotate(rotated[idx], images[i], ROTATE_180);
				shifts[i] = boxes[idx].br();
			}
		}

		Moments data = moments(img);
		Point2d best_center(data.m10 / data.m00, data.m01 / data.m00);
		double best_val = rotationalMeasure(images, shifts, area, best_center);

		vector<Point2i> points;
		convexHull(bounds, points);
		Mat hull = Mat::zeros(img.rows, img.cols, img.type());
		fillConvexPoly(hull, points, 1);
		Mat upper = Mat::zeros(img.rows, img.cols, CV_32F);
		
		list < pair<list<pair<int, int>>, int > > scheme;	// Which pairs of indexed angles to match
		int step = accum.rows / params.degree;
		int substep = step / params.num_subdiv;
		for (int t = 1; t <= params.degree / 2; t++) {
			scheme.push_back({ list<pair<int, int>>(), 2 * t == params.degree ? params.degree / 2 : params.degree });
			int k = (params.degree * params.num_subdiv % 2 == 1) ? params.degree * params.num_subdiv : params.degree * params.num_subdiv / 2;
			for (int j = 0; j < k; j++) {
				scheme.back().first.push_back({ j * substep, (j * substep + t * step) % accum.rows });
			}
		}

		for (auto elem : scheme) {
			Mat add = Mat::ones(upper.rows, upper.cols, CV_32F);
			for (auto p : elem.first) {

				double alpha = CV_2PI * p.first / accum.rows;
				double acos = cos(alpha);
				double asin = sin(alpha);

				double beta = CV_2PI * p.second / accum.rows;
				double bcos = cos(beta);
				double bsin = sin(beta);

				vector<double> dx = overlayColumns(accum, area, p.first, p.second);
				
				Mat temp = Mat::zeros(upper.rows, upper.cols, CV_32F);
				uchar* ptr = hull.data;
				float* dst = (float*)temp.data;
				for (int y = 0; y < hull.rows; y++) {
					for (int x = 0; x < hull.cols; x++) {
						if (*ptr++) {
							double d = x * (bcos - acos) + y * (bsin - asin) + (dx.size() - 1) / 2;
							int d1 = floor(d);
							int d2 = ceil(d);
							double a = d2 - d;
							double val = a * ((d1 >= 0 && d1 < dx.size()) ? dx[d1] : 0) + (1 - a) * ((d2 >= 0 && d2 < dx.size()) ? dx[d2] : 0);
							*dst++ = val;
						}
						else {
							dst++;
						}
					}
				}
				add = min(add, temp);
			}
			upper += elem.second * add;
		}
		upper = upper / (params.degree * (params.degree - 1) / 2);

		list<pair<Point2i, double>> valid;
		float* ptr = (float*)upper.data;
		for (int y = 0; y < upper.rows; y++) {
			for (int x = 0; x < upper.cols; x++) {
				if (*ptr > best_val) {
					valid.push_back( { Point2i(x, y), *ptr } );
				}
				ptr++;
			}
		}
		valid.sort([](pair<Point2i, double> a, pair<Point2i, double> b) { return a.second > b.second; });
		
		auto p = valid.begin();
		int checked = 0;
		while (p != valid.end() && p->second > best_val) {
			checked++;
//#ifdef _DEBUG
			printf("Checking point %d of %d\n", checked, valid.size());
//#endif
			double val = rotationalMeasure(images, shifts, area, p->first);
			if (val > best_val) {
				best_val = val;
				best_center = p->first;
			}
			p++;
		}
		auto finish = chrono::high_resolution_clock::now();
		double duration = chrono::duration_cast<chrono::microseconds>(finish - start).count() / 1e6;

		vector<Point2i> moves(params.degree);
		int xmin = std::numeric_limits<int>::max();
		int xmax = std::numeric_limits<int>::min();
		int ymin = std::numeric_limits<int>::max();
		int ymax = std::numeric_limits<int>::min();
		for (int i = 0; i < params.degree; i++) {
			double alpha = i * CV_2PI / params.degree;
			int dx = round(-best_center.x * cos(alpha) - best_center.y * sin(alpha) - shifts[i].x);
			int dy = round(+best_center.x * sin(alpha) - best_center.y * cos(alpha) - shifts[i].y);
			moves[i] = Point2i(dx, dy);
			xmin = min(xmin, dx);
			xmax = max(xmax, images[i].cols + dx);
			ymin = min(ymin, dy);
			ymax = max(ymax, images[i].rows + dy);
		}

		Mat red = Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_32F);
		Mat green = Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_32F);
		Mat blue = Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_32F);
		for (int i = 0; i < params.degree; i++) {
			Rect roi(moves[i].x - xmin, moves[i].y - ymin, images[i].cols, images[i].rows);
			Mat img_float;
			Mat temp = Mat::zeros(red.rows, red.cols, CV_32F);
			images[i].convertTo(img_float, CV_32F);
			img_float.copyTo(temp(roi));
			red   += 255 * temp;
			green += 255 * temp;
			blue  += 255 * temp;
		}
		Mat temp, canvas;
		merge(vector<Mat>({ red, green, blue }), temp);
		temp = 255 - (temp / params.degree);
		temp.convertTo(canvas, CV_8UC3);

		circle(canvas, best_center + (Point2d(moves[0]) - Point2d(xmin, ymin)), 3, Scalar(0, 0, 255), -1);
		printf("Mean intersection rate\t\t%f\nDegree of symmetry\t\t%d\nCenter of symmetry\t\t(%f, %f)\nPoints checked\t\t\t%d\nElapsed time (seconds)\t\t%f\n",
			best_val, params.degree, best_center.x, best_center.y, checked, duration);
#ifdef _DEBUG
		imshow("Rotational symmetry", canvas);
		waitKey(0);
#endif

		return 0;
	}

	/* ========================================================================================= */

	vector<Mat> rotated;
	vector<Rect> boxes;
	Mat accum = radon(img, bounds, params.num_angles, false, rotated, boxes);
	int davg = (accum.cols - 1) / 2;

	Point2d center;
	symmetryAxis best = firstApproach(img, area, center, params.allow_shear, bounds);
	double first_val = best.val;
	list<int> free_angles;
	vector<int> dmin(params.num_angles);
	vector<int> dmax(params.num_angles);
	vector<int> rot_area(params.num_angles);

	bool even = params.num_angles % 2 == 0;
	int angle_shift = params.num_angles / 2;
	for (int i = 0; i < params.num_angles; i++) {
		free_angles.push_back(i);
		dmin[i] = boxes[i].tl().x + davg;
		dmax[i] = boxes[i].br().x + davg;
		rot_area[i] = (!even || i < angle_shift) ? sum(accum.row(i))(0) : rot_area[i - angle_shift];
	}

	list<pair<Vec2i, double>> candidates;
	while (!free_angles.empty()) {

		auto best_iter = free_angles.begin();
		int ang_idx = *best_iter;
		free_angles.erase(best_iter);
		list<pair<Vec2i, double>> new_cands;

		// printf("Processing angle %d of %d\n", params.num_angles - free_angles.size(), params.num_angles);

		vector<int> xproj(dmax[ang_idx] - dmin[ang_idx] + 1);
		vector<int> cumsum(xproj.size() + 1);
		cumsum[0] = 0;
		for (int j = 0; j <= dmax[ang_idx] - dmin[ang_idx]; j++) {
			xproj[j] = accum.at<int32_t>(ang_idx, dmin[ang_idx] + j);
			cumsum[j + 1] = cumsum[j] + xproj[j];
		}

		int res = 0;
		for (int j = 0; j < 2 * cumsum.size() - 3; j++) {
			int idx = (j + 1) / 2;

			// 0-dimensional check
			if (j % 2 == 0) {
				res = xproj[idx] + 2 * min<int>(cumsum[idx], rot_area[ang_idx] - xproj[idx] - cumsum[idx]);
			}
			else {
				res = 2 * min<int>(cumsum[idx], rot_area[ang_idx] - cumsum[idx]);
			}

			if (res / double(rot_area[ang_idx]) > best.val) {
				// 1-dimensional check
				if (j % 2 == 0) {
					res = xproj[idx];
					for (int t = 1; t < min<int>(idx, xproj.size() - 1 - idx); t++) {
						res += 2 * min(xproj[idx - t], xproj[idx + t]);
					}
				}
				else {
					res = 0;
					for (int t = 0; t < min<int>(idx, xproj.size() - idx); t++) {
						res += 2 * min(xproj[idx - 1 - t], xproj[idx + t]);
					}
				}
				double val = res / double(rot_area[ang_idx]);
				if (val > best.val) {
					new_cands.push_back({ Vec2i(ang_idx, j), val });
				}
			}
		}
		
		if (!params.allow_shear && (new_cands.size() > 0)) 
		{
			//third_check_old(rotated, ang_idx, params, new_cands, rot_area, best);
			for (int i = 1; i <= params.num_checks; i++) {
				third_filter(i * CV_PI / (2 * params.num_checks + 2), new_cands, dmin, dmax, accum, rot_area, best.val);
			}
		}
		candidates.splice(candidates.end(), new_cands);
	}

	vector< vector< list<Vec2i>> > stripes(params.num_angles);
	vector<Mat> flipped(params.num_angles);
	candidates.sort([](pair<Vec2i, double> a, pair<Vec2i, double> b) { return a.second > b.second; });

	int num_checked_lines = 0;
	int num_best_line = 0;
	auto cand = candidates.begin();
	while (cand != candidates.end() && cand->second > best.val) {

		num_checked_lines++;
		int i = cand->first[0];
		int j = cand->first[1];

		Mat rot_dir;
		if (even) {
			rot_dir = rotated[(i + angle_shift) % params.num_angles];
			if (rot_dir.empty()) {
				rotate(rotated[i], rotated[i + angle_shift], ROTATE_90_CLOCKWISE);
				rot_dir = rotated[i + angle_shift];
			}
		}
		else {
			rot_dir = rotated[i];
		}
		Mat rot_alt = flipped[i];

		if (!params.allow_shear) {
			if (!params.use_matmul) {

				#ifdef HAVE_CUDA
					if (!params.use_cuda) {
						if (rot_alt.empty()) {
							flip(rot_dir, flipped[i], even ? 0 : 1);
							rot_alt = flipped[i];
						}
					}
					else {
						if (rot_alt.empty()) {
							cv::cuda::flip(rot_dir, flipped[i], even ? 0 : 1);
							rot_alt = flipped[i];
						}
					}
				#else
					if (rot_alt.empty()) {
						flip(rot_dir, flipped[i], even ? 0 : 1);
						rot_alt = flipped[i];
					}
				#endif

			}
			else {
					
				#ifdef HAVE_CUDA
					// CORRECT THIS!!!
					if (!params.use_cuda) {
						Mat temp;
						rot_dir.convertTo(temp, CV_32F);
						gemm(temp, temp, 1, Mat(), 0, rot_alt, GEMM_1_T);
					}
					else {
						Mat rot_temp;
						rot_dir.convertTo(rot_temp, CV_32F);
						cv::cuda::GpuMat temp(rot_temp);
						cv::cuda::gemm(temp, temp, 1, Mat(), 0, rot_alt, GEMM_1_T);
					}
				#else
					if (flipped[i].empty()) {
						Mat temp;
						rot_dir.convertTo(temp, CV_32F);
						gemm(temp, temp, 1, Mat(), 0, flipped[i], even ? GEMM_2_T : GEMM_1_T);
					}
				#endif
			}
		}
		else {
			if (stripes[i].empty()) {
				if (!even) {
					Mat rot_tmp;
					rotate(rot_dir, rot_tmp, ROTATE_90_CLOCKWISE);
					collectStripes(rot_tmp, stripes[i]);
				}
				else {
					collectStripes(rot_dir, stripes[i]);
				}
			}
		}

		if (!params.allow_shear) {
			int jmin = getMinIndex(j, even ? rot_dir.rows : rot_dir.cols);
			int res = (j % 2 == 0) ? accum.at<int32_t>(i, dmin[i] + j / 2) : 0;
			/*
				if (j % 2 == 0) 
					res = even ? countNonZero(rot_dir(Rect(0, j / 2, rot_dir.cols, 1))) :
								 countNonZero(rot_dir(Rect(j / 2, 0, 1, rot_dir.rows)));
			*/
			if (!params.use_matmul) {
				int width = (j - 1) / 2 - jmin + 1;
				Rect roi_dir = even ? Rect(0, jmin, rot_dir.cols, width) : Rect(jmin, 0, width, rot_dir.rows);
				Rect roi_alt = even ? Rect(0, rot_dir.rows - 1 - (j - jmin), rot_dir.cols, width) :
					                  Rect(rot_dir.cols - 1 - (j - jmin), 0, width, rot_dir.rows);
				Mat both;

				#ifdef HAVE_CUDA
					if (!params.use_cuda) {
						bitwise_and(rot_dir(roi_dir), rot_alt(roi_alt), both);
					}
					else {
						cv::cuda::bitwise_and(rot_dir(roi_dir), rot_alt(roi_alt), both);
					}
				#else
					bitwise_and(rot_dir(roi_dir), rot_alt(roi_alt), both);
				#endif

				res += 2 * countNonZero(both);
			}
			else {
				for (int t = jmin; t <= (j - 1) / 2; t++) {
					res += 2 * flipped[i].at<float>(t, j - t);
				}
			}
			if (res / double(rot_area[i]) > best.val) {
				num_best_line = num_checked_lines;
				best.val = res / double(rot_area[i]);
				best.rho = j / 2.0 + (dmin[i] - davg);
				best.theta = i * (CV_PI / params.num_angles);
			}
		}
		else {
			pair<double, int> res = alignStripes(stripes[i], j);
			if (res.second / double(rot_area[i]) > best.val) {
				num_best_line = num_checked_lines;
				best.val = res.second / double(rot_area[i]);
				best.rho = j / 2.0 + (dmin[i] - davg);
				best.theta = i * (CV_PI / params.num_angles);
				best.phi = res.first;
			}
		}
		cand++;
	}

	auto finish = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(finish - start).count() / 1e6;

	double rate = reflectionalMeasure(img, center, Vec3d(best.rho, best.theta, best.phi), params.visualize) / double(area);
	printf("Elapsed time:\t\t\t\t%f seconds\nSlope angle:\t\t\t\t%f degrees\nIntercept:\t\t\t\t%f pixels\n",
		   duration, best.theta * 180 / CV_PI, best.rho);
	if (best.phi > -CV_PI) {
		printf("Deviation angle:\t\t\t%f degrees\n", 180 * best.phi / CV_PI);
	}
	printf("Rotated rate (Jaccard / Absolute):\t%f / %f\nOriginal rate (Jaccard / Absolute):\t%f / %f\n",
		best.val / (2 - best.val), best.val, rate / (2 - rate), rate);

	cout << endl << "INITIAL_VALUE: " << first_val / (2 - first_val) << endl << 
					"NUM_CANDIDATES: " << candidates.size() << endl <<
		            "NUM_CHECKED_LINES: " << num_checked_lines << endl <<
		            "NUM_BEST_LINE: " << num_best_line << endl;
} 