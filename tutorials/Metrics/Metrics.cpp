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

double distanceError(vector<vector<double>> ground_truth, vector<vector<double>> my_results)
{
    if (ground_truth.size() != my_results.size())
    {
        cout<<"Something wrong with metric1's vec size!!"<<endl;
    }
    
    int N = ground_truth.size(); //N is the number of testing trajectories
    double sum = 0;
    for (int i=0; i<N; i++)
    {
        Eigen::Vector3d x_i (my_results[i][0], my_results[i][1], my_results[i][2]); //final position of trajectory i
        Eigen::Vector3d x_gs(ground_truth[i][0], ground_truth[i][1], ground_truth[i][2]); //final position of corresponding ground truth
        Eigen::Vector3d result = x_i - x_gs;
        //double vec_norm = sqrt(pow(result[0], 2) + pow(result[1], 2) + pow(result[2],2));
        sum += result.norm();
    }
    sum /= N;
    return sum;
}

double orientationError(vector<Eigen::Quaterniond> q_gs_vec, vector<Eigen::Quaterniond> q_i_vec)
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

double impulseError(vector<vector<double>> gs_vec, vector<vector<double>> my_vec)
{
    if(gs_vec.size() != my_vec.size())
    {
        cout<<"There is something wrong with metric3's vec size!!"<<endl;
    }
    int N = gs_vec.size();
    double sum = 0;
    for(int i = 0; i < N; i++)
    {
        Eigen::Vector6d imp_gs;
        imp_gs<<gs_vec[i][0], gs_vec[i][1], gs_vec[i][2], gs_vec[i][3], gs_vec[i][4], gs_vec[i][5];
        
        Eigen::Vector6d imp_i;
        imp_i<<my_vec[i][0], my_vec[i][1], my_vec[i][2], my_vec[i][3], my_vec[i][4], my_vec[i][5];
        
        Eigen::Vector6d result = imp_i - imp_gs;
        sum += result.norm();
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
    double totalLinearPositionGS[] = {}; // x y z
    double totalLinearPositionMy[] = {}; // x y z
    vector<vector<double>> linearPosGS;
    vector<vector<double>> linearPosMy;
    for (int i=0; i < sizeof(totalLinearPositionGS)/sizeof(double); i+=3)
    {
        double indiGs[] = {totalLinearPositionGS[i], totalLinearPositionGS[i+1], totalLinearPositionGS[i+2]};
        double indiMy[] = {totalLinearPositionMy[i], totalLinearPositionMy[i+1], totalLinearPositionMy[i+2]};

        vector<double> vec_GS (indiGs, indiGs+sizeof(indiGs)/sizeof(double));
        vector<double> vec_my (indiMy, indiMy+sizeof(indiMy)/sizeof(double));

        linearPosGS.push_back(vec_GS);
        linearPosMy.push_back(vec_my);
    }
    double dist_error = distanceError(linearPosGS, linearPosMy);
    cout<<"Distance error of the data set is: "<<dist_error<<endl;
    
    
    // Deal with Third metric -- impulse error
    double totalImpGS[] = {};
    double totalImpMy[] = {};
    vector<vector<double>> ImpGSFinal;
    vector<vector<double>> ImpMyFinal;
    for (int i=0; i < sizeof(totalImpGS)/sizeof(double); i+=6)
    {
        double indiGs[] = {totalImpGS[i], totalImpGS[i+1], totalImpGS[i+2], totalImpGS[i+3], totalImpGS[i+4], totalImpGS[i+5]};
        double indiMy[] = {totalImpMy[i], totalImpMy[i+1], totalImpMy[i+2], totalImpMy[i+3], totalImpMy[i+4], totalImpMy[i+5]};

        vector<double> vec_gs (indiGs, indiGs+sizeof(indiGs)/sizeof(double));
        vector<double> vec_my (indiMy, indiMy+sizeof(indiMy)/sizeof(double));

        ImpGSFinal.push_back(vec_gs);
        ImpMyFinal.push_back(vec_my);
    }
    double imp_error = impulseError(ImpGSFinal, ImpMyFinal);
    cout<<"The impulse error of data sets is: "<<imp_error<<endl;
    
    

    // Deal with second metric -- orientation error
    double quatGsTotal[] = {};//order in w x y z
    double quatMyTotal[] = {};//same order
    vector<Eigen::Quaterniond> quatGSFinal;
    vector<Eigen::Quaterniond> quatMyFinal;
    for (int i=0; i < sizeof(quatGsTotal)/sizeof(double); i+=4)
    {
        Eigen::Quaterniond q_gs (quatGsTotal[i], quatGsTotal[i+1], quatGsTotal[i+2], quatGsTotal[i+3]);
        Eigen::Quaterniond q_my (quatMyTotal[i], quatMyTotal[i+1], quatMyTotal[i+2], quatMyTotal[i+3]);
        quatGSFinal.push_back(q_gs);
        quatMyFinal.push_back(q_my);
    }
    double orientation_error = orientationError(quatGSFinal, quatMyFinal);
    cout<<"The orientation error is: "<<orientation_error<<endl;
}
