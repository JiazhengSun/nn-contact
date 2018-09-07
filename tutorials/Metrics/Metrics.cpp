#include "dart/dart.hpp"
#include "dart/gui/gui.hpp"
#include "dart/collision/bullet/bullet.hpp"
#include "dart/collision/ode/ode.hpp"
#include <dart/utils/utils.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <typeinfo>
#include <math.h>
#include <stdio.h>

using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart;
using namespace utils;
using namespace std;



//const unsigned int testSize = 61884;// 127008

class MyWindow : public dart::gui::SimWindow
{
public:
    
    MyWindow(const WorldPtr& world)
    {
        setWorld(world);
    }
    
    void keyboard(unsigned char key, int x, int y) override
    {
        switch(key)
        {
            default:
                SimWindow::keyboard(key, x, y);
        }
    }
    
    void drawWorld() const override
    {
        // Make sure lighting is turned on and that polygons get filled in
        glEnable(GL_LIGHTING);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        
        SimWindow::drawWorld();
    }
    
    void timeStepping() override
    {
        SimWindow::timeStepping();
    }
protected:
};

public double distanceError(vector<vector<double>> ground_truth; vector<vector<double>> my_results)
{
    if (ground_truth.size() != my_results.size())
    {
        cout<<"Something wrong with metric1's vec size!!"<<endl;
    }
    
    int N = ground_truth.size(); //N is the number of testing trajectories
    double sum = 0;
    for (int i=0; i<N; i++)
    {
        vector<double> x_i = my_results[i]; //final position of trajectory i
        vector<double> x_gs = ground_truth[i]; //final position of corresponding ground truth
        vector<double> result = x_i - x_gs;
        double vec_norm = sqrt(pow(result[0], 2) + pow(result[1], 2) + pow(result[2],2));
        sum += vec_norm;
    }
    sum /= N;
    return sum;
}

public double orientationError(vector<Eigen::Quaterniond> q_gs_vec, vector<Eigen::Quaterniond> q_i_vec)
{
    if(q_gs_vec.size() != q_i_vec.size())
    {
        cout<<"Something wrong with metric2's vec size!!"<<endl;
    }
    
    int N = q_gs_vec.size();
    double sum = 0;
    for (int i=0; i<N; i++)
    {
        Eigen::Quaterniond q_gs = q_gs_vec[i];
        Eigen::Quaterniond q_i = q_i_vec[i];
        Eigen::Quaterniond invert_qgs = q_gs.inverse();
        double sc = (invert_qgs * q_i).w();
        sum += 2 * acos(sc);
    }
    sum /= N;
    return sum;
}

public double impulseError(vector<vector<double>> gs_vec, vector<vector<double>> my_vec)
{
    if(gs_vec.size() != my_vec.size())
    {
        cout<<"There is something wrong with metric3's vec size!!"<<endl;
    }
    int N = gs_vec.size();
    double sum = 0;
    for(int i = 0; i < N; i++)
    {
        vector<double> imp_gs = gs_vec[i];
        vector<double> imp_i = my_vec[i];
        vector<double> result = imp_i - imp_gs;
        double norm = sqrt(pow(result[0], 2) + pow(result[1], 2) + pow(result[2],2) + pow(result[3], 2) + pow(result[4], 2) + pow(result[5],2));
        sum += norm;
    }
    sum /= N;
    return sum;
}

int main(int argc, char* argv[])
{

    WorldPtr world = SkelParser::readWorld(DART_DATA_PATH"/NN-contact-force/skel/singleBody_genData.skel");
    assert(world != nullptr);
    world->setGravity(Eigen::Vector3d(0.0, 0.0, 0));

    //Deal with first metric -- distance error
    double totalLinearPositionGS[] = {};
    double totalLinearPositionMy[] = {};
    vector<vector<double>> linearPosGS;
    vector<vector<double>> linearPosMy;
    for (int i=0; i < totalLinearPositionGS.size(); i+=3)
    {
        double indiGs[] = {totalLinearPositionGS[i], totalLinearPositionGS[i+1], totalLinearPositionGS[i+2]};
        double indiMy[] = {totalLinearPositionMy[i], totalLinearPositionMy[i+1], totalLinearPositionMy[i+2]};
        
        vector<double> vec_GS (indiGs, indiGs+sizeof(indiGs)/sizeof(double));
        vector<double> vec_my (indiMy, indiMy+sizeof(indiMy)/sizeof(double));
        
        linearPosGS.push_back(vec_GS);
        linearPosMy.push_back(vec_my)
    }
    double dist_error = distanceError(linearPosGS, linearPosMy);
    
    // Deal with Third metric -- impulse error
    double totalImpGS[] = {};
    double totalImpMy[] = {};
    vector<vector<double>> ImpGSFinal;
    vector<vector<double>> ImpMyFinal;
    for (int i=0; i < totalImpGS.size(); i+=6)
    {
        double indiGs[] = {totalImpGS[i], totalImpGS[i+1], totalImpGS[i+2], totalImpGS[i+3], totalImpGS[i+4], totalImpGS[i+5]};
        double indiMy[] = {totalImpMy[i], totalImpMy[i+1], totalImpMy[i+2], totalImpMy[i+3], totalImpMy[i+4], totalImpMy[i+5]};
        
        vector<double> vec_gs (indiGs, indiGs+sizeof(indiGs)/sizeof(double));
        vector<double> vec_my (indiMy, indiMy+sizeof(indiMy)/sizeof(double));
        
        ImpGSFinal.push_back(vec_gs);
        ImpMyFinal.push_back(vec_my)
    }
    double dist_error = distanceError(ImpGSFinal, ImpMyFinal);
    
    // Deal with second metric -- orientation error
    double quatGsTotal[] = {};//order in w x y z
    double quatMyTotal[] = {};//same order
    vector<Eigen::Quaterniond> quatGSFinal;
    vector<Eigen::Quaterniond> quatMyFinal;
    for (int i=0; i < quatGsTotal.size(); i+=4)
    {
        Eigen::Quaterniond q_gs (quatGsTotal[i], quatGsTotal[i+1], quatGsTotal[i+2], quatGsTotal[i+3]);
        Eigen::Quaterniond q_my (quatMyTotal[i], quatMyTotal[i+1], quatMyTotal[i+2], quatMyTotal[i+3]);
        quatGSFinal.push_back(q_gs);
        quatMyFinal.push_back(q_my);
    }
    double orientation_error = orientationError(quatGSFinal, quatMyFinal);
    
    
//     vector<vector<double>> x_test;
//     double first_arr[] = {-1.93004, 1.93004, -0.511833, 0.692334, 0.0538576, -0.81607, 1.53919,-2.18402, 2.01191};
//     vector<double> first_vec (first_arr, first_arr + sizeof(first_arr) / sizeof(double));
//     x_test.push_back(first_vec);
//
//     double second_arr[] = {1.25728, 1.13107, -1.13107, 0.182227, 0.691963, -0.175838, 3.41511, -5.99775, -0.846054};
//     vector<double> second_vec (second_arr, second_arr + sizeof(second_arr) / sizeof(double));
//     x_test.push_back(second_vec);
//
//     double third_arr[] = {0.246902, -2.09713, 2.09713, -0.69256, 0.14331, 0.604811, -4.66946, 0.895696, -0.0151988};
//     vector<double> third_vec (third_arr, third_arr + sizeof(third_arr) / sizeof(double));
//     x_test.push_back(third_vec);
//
//     double fourth_arr[] = {-2.12687, 2.12687, -0.192541, 0.511687, -0.07551, 0.902735, 1.32739, -1.57741, 3.24697};
//     vector<double> fourth_vec (fourth_arr, fourth_arr + sizeof(fourth_arr) / sizeof(double));
//     x_test.push_back(fourth_vec);
//
//     double fifth_arr[] = {-0.235922, -0.235922, 1.55789, 0.384338, -0.43193, 0.553732, 2.83865, -2.01604, -2.1784};
//     vector<double> fifth_vec (fifth_arr, fifth_arr + sizeof(fifth_arr) / sizeof(double));
//     x_test.push_back(fifth_vec);
//
//     for (int k = 0; k < 5; k++)
//     {
//         vec_t in;
//         double* test_pr = &x_test[k].front();
//         in.assign(test_pr, test_pr + x_test[k].size());
//
//         vec_t result = net.predict(in);
//         for (int i = 0; i < result.size(); i++) {
//             cout<<result[i]<<' ';
//         }
//         cout<<' '<<endl;
//     }

    // cout<<"Debugging!"<<endl;
    // cout<<' '<<endl;
    // layer* h;
    // for (int n = 0; n < net.layer_size(); n++) {
    //     h = net[n];
    //     if (h->layer_type() == "fully-connected") {
    //         cout<<"Current at layer "<<n<<endl;
    //         auto info = h->weights();
    //         vec_t &w = *(info[0]);
    //         vec_t &b = *(info[1]);
    //         for (int k = 0; k < w.size(); k++) {
    //             cout<<w[k]<<' ';
    //         }
    //         cout<<' '<<endl;
    //     }
    // }

}
