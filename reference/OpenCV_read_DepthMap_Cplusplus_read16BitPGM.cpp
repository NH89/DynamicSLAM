
#include <iostream>
#include <fstream>
#include <opencv2/core.hpp> 
//#include <opencv2/core/core.hpp> 

using namespace std;
using namespace cv;

bool read16BitPGM(Mat &dst, const string &imgPath){
	ifstream in(imgPath.c_str(), ios::binary);
	if(!in.is_open()){
		cout << "Unable to open image file \'" << imgPath << '\'' << endl;
		return false;
	}
	//read header
	string magic;
	unsigned int height, width, maxVal;
	in >> magic;
	in >> height;
	in >> width;
	in >> maxVal;
	in.ignore(64, '\n');
	const unsigned int nBytes = 2*width*height;
	in.read((char*)dst.data, nBytes);
	in.close();

	return true;
}
