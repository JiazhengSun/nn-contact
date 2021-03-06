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
#include <cmath>

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
        Eigen::Vector3d result = x_i - x_gs; // The real metric
//        Eigen::Vector3d result = x_gs; // For average
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
        
//        Eigen::Quaterniond q_i;
//        q_i.w()=1; q_i.x()=0; q_i.y()=0; q_i.z()=0;
        
        Eigen::Quaterniond invert_qgs = q_gs.inverse();
        double temp = (invert_qgs * q_i).w();
        if (temp > 1.0 or temp < -1.0)
        {
            cout<<"out of range value is: "<<temp<<endl;
            if (temp > 1) {temp = 1.0;}
            if (temp < -1) {temp = -1.0;}
        }
        double sc = abs(temp);
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
        
//        Eigen::Vector6d result = imp_gs;
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
    double totalLinearPositionGS[] =
    {0.161964, 0.0499987, 9.77873e-18,
        0.51383, 0.0489933, 4.32709e-19,
        0.857411, 0.0490489, -6.76834e-18,
        0.342627, 0.0489119, 4.76551e-18,
        -0.171164, 0.0485047, 2.18556e-17,
        1.43569, 0.0493664, -1.77775e-16,
        -0.0784273, 0.0499997, 1.64451e-18,
        -0.797167, 0.0490728, 7.83664e-18,
        0.613375, 0.049534, -1.39767e-19,
        -0.860755, 0.0498612, 5.13575e-18,
        0.643162, 0.0498881, 3.58722e-18,
        -0.682734, 0.0478772, 2.59889e-18,
        -0.967236, 0.048508, 3.04634e-17,
        -0.0593886, 0.0495062, 2.8301e-18,
        2.34278, 0.0493406, -4.08309e-16,
        4.36817, 0.105432, 1.7402e-14,
        1.65901, 0.0492206, -3.1972e-17,
        -0.630024, 0.0467572, -5.51925e-18,
        1.5284, 0.0493787, 1.22854e-16,
        1.23329, 0.0499634, 5.44297e-18,
        0.869696, 0.0499976, -2.10031e-17,
        1.86832, 0.0499944, -1.06641e-16,
        -0.493497, 0.0499969, 1.57098e-18,
        -0.356992, 0.0491546, 2.93668e-17,
        0.245343, 0.0499982, 1.66609e-18,
        -1.40354, 0.0488898, 1.27176e-16,
        0.104251, 0.0491218, -4.62024e-18,
        0.231074, 0.0999995, -1.8688e-18,
        -1.21352, 0.0496559, -2.28575e-16,
        -1.40196, 0.0490957, -4.11024e-17,
        -1.6262, 0.0499872, 1.12847e-17,
        -1.43012, 0.0999933, 6.10021e-17,
        -1.17412, 0.0463415, 8.10648e-19,
        0.0420654, 0.0497055, 1.50376e-18,
        0.27628, 0.0473628, 2.03274e-18,
        1.47062, 0.0467924, -9.93295e-18,
        -0.526691, 0.0481742, 3.40495e-18,
        -1.04078, 0.0459685, -1.1923e-18,
        -1.34293, 0.0498541, 3.25203e-17,
        -0.754341, 0.046943, -1.54309e-19,
        -2.17782, 0.0491012, 1.76586e-15,
        1.22307, 0.048221, -3.85015e-18,
        -1.94528, 0.0495751, 3.43542e-16,
        -0.492307, 0.0488663, 5.49713e-18,
        -0.42734, 0.04924, 3.89051e-18,
        0.49263, 0.0474317, -6.92331e-19,
        -1.34587, 0.0492301, -1.44092e-17,
        0.190466, 0.0471869, 5.83609e-20,
        3.27237, 0.049212, -4.01613e-16,
        0.502099, 0.0463139, -3.71134e-18,
        2.018, 0.0499783, 4.49288e-16,
        -1.55435, 0.0494919, -2.25505e-16,
        -0.265815, 0.0488777, -2.2539e-18,
        0.993678, 0.0488078, 2.09582e-19,
        1.76161, 0.048585, -1.97429e-16,
        0.639857, 0.0485693, -1.03345e-17,
        -1.71661, 0.0498364, -3.33846e-16,
        0.0754411, 0.0497107, -3.12601e-18,
        -1.04572, 0.0499169, -5.89611e-18,
        0.958697, 0.0470378, -1.24274e-17,
        -0.347825, 0.1, 1.04522e-18,
        0.447072, 0.0978626, 7.95958e-19,
        1.09812, 0.0496135, 3.20443e-16,
        0.576194, 0.0499993, 2.82783e-17,
        0.466915, 0.0479754, -4.39318e-18,
        0.541577, 0.0499784, -4.06984e-19,
        -1.15872, 0.0478881, 3.37554e-17,
        -0.384886, 0.0466081, -8.42128e-19,
        0.856857, 0.0445214, 3.48925e-19,
        -0.811663, 0.0486827, 9.19444e-18,
        -0.660248, 0.0491243, 7.94559e-18,
        0.190099, 0.1, 6.27608e-19,
        2.04975, 0.0490897, -5.11065e-17,
        -0.703087, 0.0472989, 2.19449e-18,
        0.601704, 0.0478731, -4.90026e-20,
        -1.58798, 0.0472906, 1.95556e-17,
        0.452812, 0.0476439, -3.0814e-18,
        0.248497, 0.0498702, 7.48911e-19,
        -1.1669, 0.0999973, 1.12045e-16,
        0.85914, 0.0490957, -9.93981e-19,
        -1.58958, 0.0489654, 2.63219e-16,
        0.101095, 0.0498943, 1.10615e-19,
        -0.337414, 0.0485047, 1.38276e-18,
        -0.737336, 0.0498043, 3.44104e-17,
        0.629407, 0.0490369, -1.98228e-18,
        0.601386, 0.0498462, -3.24205e-18,
        -0.619553, 0.0499972, 3.20293e-17,
        -0.193732, 0.0476578, 1.29594e-18,
        0.804235, 0.0499915, -3.35163e-18,
        1.12775, 0.0499973, 2.24705e-17,
        0.179509, 0.0491461, -1.55547e-18,
        -1.04993, 0.0488298, 2.20974e-16,
        1.77281, 0.0489474, 3.49469e-16,
        0.527981, 0.0497868, -1.8357e-17,
        2.13984, 0.0494417, 3.75e-16,
        -0.850325, 0.0487528, 7.21555e-18,
        -0.934014, 0.0483915, 3.99486e-18,
        -0.379591, 0.0489967, 2.85349e-18,
        -1.26512, 0.0471372, 6.42678e-18,
        -0.266903, 0.0999999, -9.54102e-19}; // x y z
    
    double totalLinearPositionMy[] =
    {0.177507, 0.0486046, 0,
        0.554087, 0.0435932, 0,
        0.858415, 0.0478941, 0,
        0.355954, 0.0490109, 0,
        -0.134539, 0.0494741, 0,
        1.39048, 0.0483897, 0,
        -0.0555319, 0.0477756, 0,
        -0.794329, 0.0472458, 0,
        0.460601, 0.0993229, 0,
        -0.846152, 0.0479076, 0,
        0.65519, 0.0484089, 0,
        -0.683235, 0.0453515, 0,
        -1.07893, 0.0492317, 0,
        -0.0612982, 0.0466117, 0,
        2.38908, 0.0491188, 0,
        4.11655, 0.0488469, 0,
        1.77765, 0.0490618, 0,
        -0.636099, 0.0432891, 0,
        1.54941, 0.0486604, 0,
        1.25045, 0.0440624, 0,
        0.837832, 0.0486808, 0,
        1.94494, 0.0998895, 0,
        -0.560572, 0.0495297, 0,
        -0.29556, 0.0464921, 0,
        0.337473, 0.0494725, 0,
        -1.42621, 0.0489957, 0,
        0.104233, 0.0480991, 0,
        0.23267, 0.0999883, 0,
        -1.28351, 0.0998589, 0,
        -1.33392, 0.0444306, 0,
        -2.13446, 0.0475109, 0,
        -1.40784, 0.0489044, 0,
        -1.17449, 0.045106, 0,
        0.0487037, 0.0497387, 0,
        0.277624, 0.0442967, 0,
        1.46584, 0.0462216, 0,
        -0.526198, 0.0433314, 0,
        -1.04094, 0.0439216, -2.76333e-16,
        -1.32351, 0.0491943, 0,
        -0.754532, 0.0455752, 0,
        -2.34249, 0.0476538, 0,
        1.21702, 0.0473589, 0,
        -2.08149, 0.0486113, 0,
        -0.461714, 0.0493626, 0,
        -0.403768, 0.049411, 0,
        0.492088, 0.046293, 0,
        -1.41338, 0.0470641, 0,
        0.256397, 0.0488028, 0,
        2.28026, 0.0495506, -5.7582e-16,
        0.502433, 0.0442129, 0,
        2.39587, 0.0482098, 0,
        -1.47274, 0.0464327, 0,
        -0.217739, 0.0490115, 0,
        0.993522, 0.0476044, -3.13338e-16,
        1.93287, 0.0487297, 0,
        0.6609, 0.0496606, 0,
        -1.67886, 0.0485935, 0,
        0.0878336, 0.0479977, 0,
        -0.929837, 0.0468234, 0,
        0.958635, 0.0496171, 0,
        -0.345435, 0.0999743, 0,
        0.44627, 0.0962159, 0,
        1.05658, 0.0495894, 0,
        0.587749, 0.0487467, 0,
        0.46702, 0.0460472, 0,
        0.541791, 0.0481505, 0,
        -1.15237, 0.0998885, 0,
        -0.385321, 0.0448018, 0,
        0.856423, 0.0431233, 0,
        -0.809551, 0.0485517, 0,
        -0.64255, 0.048314, 0,
        0.190686, 0.0999636, 0,
        2.04545, 0.0492975, 0,
        -0.68449, 0.0488593, 0,
        0.601341, 0.0463513, 0,
        -1.42063, 0.0490541, 0,
        0.454151, 0.0477105, 0,
        0.28971, 0.0466384, 0,
        -1.21332, 0.0998762, 0,
        0.858072, 0.0465478, 0,
        -1.47215, 0.0487784, 0,
        0.147872, 0.0490918, 0,
        -0.351589, 0.0455233, 0,
        -0.70012, 0.049368, 0,
        0.586255, 0.0998674, 0,
        0.602154, 0.0475217, 0,
        -0.695538, 0.0998804, 0,
        -0.15656, 0.0371906, 0,
        0.778718, 0.0486313, 0,
        1.16919, 0.0489074, 0,
        0.220018, 0.0469624, 0,
        -0.887501, 0.0466197, 0,
        1.76078, 0.0999139, 0,
        0.534575, 0.0495291, 0,
        2.28579, 0.048366, 0,
        -0.850349, 0.0453349, 0,
        -0.932833, 0.0466368, 0,
        -0.391517, 0.0463701, 0,
        -1.26343, 0.047497, 0,
        -0.267787, 0.0999213, 0}; // x y z

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
    double totalImpGS[] =
    {-1.25259e-17, 3.19906e-17, -0.0642509, -2.46454, -1.87478, 5.89969e-16,
        -1.38153e-17, -3.45696e-17, -0.045694, -2.20228, 0.644201, -3.07965e-16,
        1.92937e-17, -3.77405e-17, 0.0629567, 1.6944, -1.47677, -4.15532e-16,
        5.34936e-18, 8.60007e-18, -0.073467, -0.352184, -0.910762, 1.80014e-16,
        3.23035e-17, -2.15705e-18, -0.0135575, 2.43513, 1.35314, 4.98985e-16,
        1.45061e-17, -1.86038e-17, -0.119927, -2.25223, -2.32538, 1.86107e-16,
        7.11855e-18, -1.42371e-17, -0.00662859, 2.09709, 1.11483, 4.28852e-16,
        -2.88739e-17, -3.24578e-17, -0.0902821, -2.57883, -2.19224, 4.89224e-16,
        1.89754e-18, -1.00827e-17, -0.00686825, 3.49654, -1.67959, -2.04327e-16,
        5.84961e-19, 1.27079e-17, -0.0458324, 3.29337, 2.10501, 1.57605e-16,
        7.45054e-18, -3.02017e-17, -0.0759104, -3.28361, 0.882699, -8.55865e-16,
        2.76789e-17, 1.75007e-17, -0.102297, 1.82747, 1.93671, 6.17312e-16,
        -8.58038e-18, -3.28296e-18, 0.106987, -1.58233, 1.86103, 1.19706e-16,
        -2.25453e-17, -2.77434e-17, 0.084821, 1.53778, -1.6171, 4.03299e-16,
        3.09718e-18, 2.87458e-17, -0.0656391, -3.96292, 1.32507, -2.63949e-16,
        -6.90634e-18, 5.26927e-18, -0.104525, -3.14145, 0.525479, -3.33802e-16,
        1.31283e-18, 1.8191e-17, -0.0946995, -1.05742, -1.47571, 3.84167e-16,
        1.85651e-18, -3.65956e-18, 0.110056, 0.581626, -1.39137, -3.60745e-16,
        4.32192e-17, -1.07061e-17, -0.0608955, 4.11646, -1.44927, 2.13365e-16,
        -3.10521e-17, -2.04709e-17, -0.053014, -4.23072, 1.58522, -5.6619e-16,
        -8.55679e-18, 1.27768e-17, -0.0212425, -2.1027, -1.26377, -1.31379e-16,
        7.23941e-18, 2.83566e-17, -0.0269565, 4.12979, -1.79533, 4.92146e-17,
        9.87314e-18, -4.00209e-17, 0.0665697, 2.2464, 0.457504, 5.91585e-16,
        2.31393e-17, -1.61714e-17, 0.041081, 2.36799, -1.59481, 3.22021e-16,
        -6.26217e-19, -2.25028e-17, -0.0300408, 2.37185, -0.885517, 1.80794e-16,
        -6.1655e-18, -5.01624e-18, 0.102829, -2.79743, -0.370429, -2.75298e-16,
        -2.35707e-18, 2.65109e-17, 0.119284, 1.7539, -2.06979, -5.77826e-16,
        -1.304e-17, -9.26569e-19, -0.0276263, 3.8378, -1.64264, -8.31615e-16,
        4.60112e-18, 1.78759e-17, 0.0261515, -3.65659, 2.08981, -6.62718e-16,
        -3.06459e-17, -1.27196e-17, 0.0860375, 3.68203, -2.70139, -1.51334e-16,
        -1.95387e-17, -1.64337e-17, 0.122547, -1.26044, 0.595253, -3.19496e-18,
        -3.32725e-17, -3.14169e-17, 0.119747, -2.37463, 2.38479, -1.03304e-15,
        1.80768e-17, 8.94926e-18, -0.105892, 1.67737, 1.8976, 1.69452e-16,
        2.77078e-17, -2.76601e-17, -0.0467268, 2.01463, 1.47458, 4.10744e-16,
        -3.4609e-17, -1.40488e-17, -0.111753, 1.16553, 1.70029, 1.4023e-16,
        -2.45507e-18, -2.02876e-18, -0.125508, -0.470065, 1.02004, 4.50698e-17,
        -1.38121e-19, 1.36015e-17, 0.0867244, 0.206129, 0.764179, 1.8206e-16,
        -3.81888e-18, 1.90321e-18, 0.0899852, -0.7521, -0.523802, 4.88583e-17,
        -1.07377e-17, -2.14755e-17, 0.0726956, -2.38399, 1.91895, -8.84593e-17,
        2.36513e-17, -1.33242e-18, -0.106025, -1.41053, -1.76552, 5.25673e-17,
        2.23549e-18, -2.35529e-17, 0.134435, -2.85667, -0.083984, -8.63849e-17,
        7.2731e-18, -4.13785e-18, -0.100353, -0.804601, 0.601225, -9.2358e-17,
        -1.92886e-18, -3.08118e-18, 0.0797371, 1.85546, 0.130358, -1.78845e-17,
        -1.54273e-17, -1.77176e-17, -0.0322722, -1.47983, -1.06264, 3.12961e-16,
        3.12481e-17, -6.98508e-18, -0.033049, -3.31979, -1.99039, -2.85862e-16,
        5.14356e-17, -1.71506e-18, -0.035973, 1.20086, -4.11744, -5.51864e-16,
        -5.17694e-18, 1.97864e-17, 0.0414169, 3.27618, 1.22392, 1.23732e-16,
        7.38482e-19, -4.36258e-17, 0.00244668, 2.99287, -1.5209, -4.20166e-17,
        1.19257e-17, -1.60472e-17, -0.127438, 2.70814, -0.0796947, -5.16986e-16,
        1.11634e-17, -3.66537e-17, 0.0619703, 1.20406, -1.22173, -3.40556e-16,
        9.89052e-18, 3.01894e-17, -0.12776, -3.85792, 0.651358, 7.92489e-17,
        2.86848e-17, -2.24278e-17, 0.0758697, -3.03596, 2.27668, -4.83978e-16,
        8.70446e-18, 2.43478e-17, -0.00830008, 3.14518, -1.48959, -5.52695e-16,
        5.134e-18, 1.0077e-18, -0.147994, 0.56629, -1.1968, -3.24141e-16,
        3.04559e-17, -1.88447e-17, -0.0902495, -1.92332, -1.86415, -1.48956e-16,
        -1.59364e-17, -1.323e-17, -0.0693627, -1.06153, -1.22439, -1.10239e-16,
        -5.48738e-18, -6.37248e-18, 0.0705059, 3.19264, 0.89126, -5.41162e-17,
        -4.69227e-17, 3.2992e-18, 0.0277884, -3.14572, 1.85074, -5.10551e-16,
        -6.9263e-18, 4.50776e-17, 0.0133405, 3.3488, 1.541, -2.42222e-16,
        -3.30702e-17, 4.14125e-17, 0.0634785, -1.49663, 1.3831, -6.39698e-16,
        9.7539e-18, -3.42686e-17, -0.0291252, 3.18143, -1.29946, 4.9598e-16,
        -1.0973e-17, -2.15123e-17, 0.0288644, 4.24695, 1.22552, 1.44588e-16,
        -1.11272e-18, -5.26121e-18, -0.0381621, -1.62892, 0.43284, -4.44056e-16,
        -1.69668e-17, 3.30663e-17, -0.0475371, 2.17238, 1.56156, -1.02451e-16,
        4.38185e-18, -2.4196e-17, 0.0496821, 0.927111, -0.960377, -2.90904e-16,
        -3.65214e-17, 1.5428e-17, 0.0825855, -1.07483, 1.36327, -2.66761e-16,
        3.0349e-18, -4.33855e-18, 0.134952, -0.364669, 1.53185, -4.20723e-18,
        -2.07979e-17, 6.75952e-18, 0.0909916, 0.746256, -1.28304, 1.59791e-16,
        1.8362e-17, 2.09042e-18, -0.11096, 0.555182, -0.83201, -2.22119e-16,
        2.30152e-17, 2.33585e-17, -0.0454783, 2.00463, 1.4571, 3.44085e-16,
        -7.92941e-18, 1.9703e-17, 0.0597508, -0.694796, 0.944906, 7.76372e-17,
        1.89447e-17, 2.45607e-17, 0.0127159, 2.34012, 1.0429, 2.10948e-16,
        3.33517e-17, 2.61042e-17, -0.0344826, 4.56906, 2.62936, 3.02169e-16,
        -2.57303e-17, 2.31583e-18, 0.120789, 0.438702, -1.42724, 1.3459e-16,
        -1.8067e-17, -1.01132e-17, 0.0808446, 1.08472, -1.35081, -5.89905e-17,
        -7.86883e-18, 3.3443e-18, 0.129818, 0.686907, -1.64164, 1.3755e-16,
        -3.96217e-18, 5.95345e-18, 0.0497695, -2.02553, 1.51046, -1.65099e-16,
        1.56689e-17, -1.84747e-18, -0.0316627, -3.17624, -1.90475, 1.49564e-16,
        2.93431e-17, -7.23334e-18, 0.0408315, 3.0912, -1.95392, -7.30892e-18,
        -1.47772e-17, -1.67062e-18, -0.140991, -0.232325, 1.29375, -9.2923e-17,
        2.54961e-17, 3.71144e-17, 0.102454, -2.73825, 2.39366, -4.13077e-16,
        1.8531e-18, 2.05799e-17, -0.0423021, 2.14839, 1.49722, -3.96871e-16,
        4.31647e-18, 1.99086e-17, 0.0295049, -2.36398, 1.47704, 7.53433e-17,
        2.15991e-18, 9.524e-18, 0.0610047, -2.57992, 1.90001, 3.49042e-16,
        1.40466e-18, -7.59902e-18, -0.0127176, 2.88159, -1.31362, -2.69996e-16,
        4.8274e-17, -4.22298e-17, 0.0252944, 2.70485, -1.60537, -9.01232e-16,
        2.08982e-17, 3.13881e-17, 0.0504804, -1.81674, 1.41317, 7.58952e-16,
        -9.96327e-18, -8.90142e-19, -0.0146807, 3.0261, 1.65985, 4.0284e-16,
        -1.18859e-17, 2.37717e-17, -0.043897, -3.22072, -2.04933, 3.49107e-16,
        3.88172e-17, 2.90621e-17, -0.045892, -3.76026, 1.42121, 3.26616e-16,
        -1.66313e-17, 2.03749e-18, -0.0363147, -2.21107, -1.46868, 1.0452e-16,
        2.53144e-17, 4.54246e-17, 0.0320152, -2.27096, 1.45563, 6.79392e-16,
        1.15394e-19, 3.254e-17, -0.0871761, 3.14189, -0.699183, -2.01559e-16,
        -9.98779e-18, -7.78e-18, -0.0566831, -2.05143, -1.59255, 3.87803e-16,
        -4.43178e-18, -3.27698e-17, -0.103031, 3.40338, 2.732, 2.75744e-16,
        -6.68071e-18, 5.72053e-18, 0.0920408, 0.226411, -1.03361, -1.2407e-17,
        3.32011e-18, 1.41765e-17, -0.0439044, 1.91061, 1.39435, 1.93125e-16,
        -1.31261e-17, -2.37066e-18, 0.0180588, -3.39933, -1.51908, -3.10215e-17,
        -1.60884e-17, -4.4151e-17, -0.0831864, -1.53061, -1.59717, 5.14779e-16,
        9.44815e-18, -1.84003e-17, 0.0129494, -2.82547, 1.54223, -4.41734e-16};

    double totalImpMy[] =
    {0, 0, -0.0592166, -2.54057, -1.86245, 0,
        0, 0, -0.0421697, -2.18571, 0.671155, 0,
        0, 0, 0.0675462, 1.63331, -1.49212, 0,
        0, 0, -0.0731062, -0.391677, -0.9269, 0,
        0, 0, -0.0162086, 2.41663, 1.3704, 0,
        0, 0, -0.108702, -2.37714, -2.27559, 0,
        0, 0, 0.00900034, 2.13604, 0.978018, 0,
        0, 0, -0.0787871, -2.59232, -2.08403, 0,
        0, 0, -0.0154762, 3.64666, -1.66857, 0,
        0, 0, -0.0364971, 3.26146, 1.9957, 0,
        0, 0, -0.0792104, -3.33231, 0.87405, 0,
        0, 0, -0.0957275, 1.79731, 1.85593, 0,
        0, 0, 0.104215, -1.75878, 1.92154, 0,
        0, 0, 0.0816383, 1.5597, -1.59623, 0,
        0, 0, -0.0671452, -3.99875, 1.32792, 0,
        0, 0, -0.105698, -3.16352, 0.524778, 0,
        0, 0, -0.0847756, -1.18992, -1.44272, 0,
        0, 0, 0.112779, 0.517125, -1.38635, 0,
        0, 0, -0.0546724, 4.07694, -1.49175, 0,
        0, 0, -0.0491787, -4.21399, 1.61521, 0,
        0, 0, -0.00802527, -1.85634, -1.00842, 0,
        0, 0, -0.0310095, 4.19276, -1.78628, 0,
        0, 0, 0.0712449, 2.34049, 0.457798, 0,
        0, 0, 0.0412628, 2.33442, -1.57984, 0,
        0, 0, -0.0105269, 2.39521, -1.09234, 0,
        0, 0, 0.101989, -2.70123, -0.330719, 0,
        0, 0, 0.120377, 1.61173, -2.00964, 0,
        0, 0, -0.0295543, 3.85278, -1.63085, 0,
        0, 0, 0.035122, -3.61227, 2.15736, 0,
        0, 0, 0.0851883, 3.72112, -2.71244, 0,
        0, 0, 0.128347, -1.28424, 0.641348, 0,
        0, 0, 0.113746, -2.43515, 2.35503, 0,
        0, 0, -0.103314, 1.65311, 1.8597, 0,
        0, 0, -0.0440966, 2.0325, 1.45722, 0,
        0, 0, -0.097113, 1.32913, 1.6357, 0,
        0, 0, -0.127694, -0.500951, 1.02647, 0,
        0, 0, 0.0885622, 0.331333, 0.719956, 0,
        0, 0, 0.0780664, -0.654774, -0.453277, 0,
        0, 0, 0.0742932, -2.32495, 1.90541, 0,
        0, 0, -0.096057, -1.39123, -1.65619, 0,
        0, 0, 0.13499, -2.71806, -0.00913107, 0,
        0, 0, -0.0992237, -0.774075, 0.605199, 0,
        0, 0, 0.076008, 1.81809, 0.148963, 0,
        0, 0, -0.0279643, -1.54407, -1.05168, 0,
        0, 0, -0.0312853, -3.10181, -1.86376, 0,
        0, 0, -0.0299984, 1.25209, -4.11967, 0,
        0, 0, 0.0357013, 3.22998, 1.25798, 0,
        0, 0, 0.0047039, 2.97311, -1.53359, 0,
        0, 0, -0.119766, 2.57703, -0.0908542, 0,
        0, 0, 0.0643272, 1.13439, -1.21047, 0,
        0, 0, -0.127663, -3.85554, 0.651137, 0,
        0, 0, 0.0674948, -2.94184, 2.14587, 0,
        0, 0, -0.0133767, 3.26649, -1.49948, 0,
        0, 0, -0.148201, 0.460147, -1.25194, 0,
        0, 0, -0.0871929, -1.9157, -1.82978, 0,
        0, 0, -0.0683827, -0.986152, -1.1769, 0,
        0, 0, 0.0618311, 3.14066, 0.952019, 0,
        0, 0, 0.0273074, -3.14894, 1.84754, 0,
        0, 0, 0.0191942, 3.32483, 1.47047, 0,
        0, 0, 0.0735769, -1.45422, 1.46288, 0,
        0, 0, -0.0237501, 3.10295, -1.31398, 0,
        0, 0, 0.0120497, 4.24837, 1.25954, 0,
        0, 0, -0.0354779, -1.58862, 0.439531, 0,
        0, 0, -0.0465281, 2.23742, 1.58399, 0,
        0, 0, 0.0520349, 0.841692, -0.941195, 0,
        0, 0, 0.0829672, -0.990245, 1.32479, 0,
        0, 0, 0.139522, -0.307574, 1.54901, 0,
        0, 0, 0.0645886, 0.778495, -1.03513, 0,
        0, 0, -0.0981331, 0.522752, -0.719955, 0,
        0, 0, -0.0407604, 2.11364, 1.46442, 0,
        0, 0, 0.0548436, -0.676003, 0.886438, 0,
        0, 0, 0.00935531, 2.38077, 1.09683, 0,
        0, 0, -0.0320151, 4.56748, 2.60389, 0,
        0, 0, 0.125229, 0.33185, -1.41821, 0,
        0, 0, 0.0758656, 0.999144, -1.25823, 0,
        0, 0, 0.13613, 0.593804, -1.65821, 0,
        0, 0, 0.0532712, -1.96389, 1.51466, 0,
        0, 0, -0.0294722, -3.24491, -1.91718, 0,
        0, 0, 0.0437129, 3.11599, -1.99512, 0,
        0, 0, -0.145288, -0.325035, 1.29036, 0,
        0, 0, 0.112039, -2.67718, 2.45898, 0,
        0, 0, -0.0344369, 2.16588, 1.42731, 0,
        0, 0, 0.0373399, -2.29774, 1.52227, 0,
        0, 0, 0.0737861, -2.53877, 2.00725, 0,
        0, 0, -0.0189658, 2.98058, -1.30063, 0,
        0, 0, 0.0245494, 2.68757, -1.58928, 0,
        0, 0, 0.0481593, -1.93078, 1.44699, 0,
        0, 0, -0.00244963, 2.9568, 1.5029, 0,
        0, 0, -0.0474111, -3.15536, -2.05179, 0,
        0, 0, -0.0417371, -3.74781, 1.45653, 0,
        0, 0, -0.0283799, -2.27171, -1.41965, 0,
        0, 0, 0.0390048, -2.23992, 1.51001, 0,
        0, 0, -0.093654, 3.17941, -0.653162, 0,
        0, 0, -0.0657041, -2.11289, -1.71348, 0,
        0, 0, -0.105977, 3.42037, 2.76995, 0,
        0, 0, 0.0975826, 0.20675, -1.0792, 0,
        0, 0, -0.0395419, 2.01929, 1.40507, 0,
        0, 0, 0.0154376, -3.29338, -1.49232, 0,
        0, 0, -0.0750743, -1.54069, -1.52109, 0,
        0, 0, 0.0283773, -2.86101, 1.71428, 0 };
    
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
    double quatGsTotal[] =
    {2.70378e-06, -5.50896e-17, -8.47225e-18, 1,
        0.0050208, -1.81785e-18, 5.96808e-18, 0.999987,
        0.00445682, 5.40619e-17, 1.80387e-17, 0.99999,
        0.00542551, -1.241e-17, 5.99617e-18, 0.999985,
        0.99999, 1.80123e-16, 3.32278e-17, -0.00457174,
        0.999995, 2.38651e-17, 4.56272e-16, -0.00310437,
        1, 9.05886e-19, -2.24669e-18, 2.1108e-07,
        0.00410728, -5.08382e-17, 4.44886e-17, -0.999992,
        0.999997, -3.71681e-18, -3.66143e-18, -0.00232701,
        1, -7.77096e-18, 3.05581e-17, 0.000654994,
        0.000464904, -1.60847e-16, -1.87807e-17, 1,
        0.999944, 8.23042e-18, 8.3632e-18, 0.0105575,
        0.999972, -6.38607e-18, -4.32649e-18, 0.00743186,
        0.0024634, -2.68871e-17, 8.09088e-18, -0.999997,
        0.00329148, 9.80798e-16, -4.78119e-17, 0.999995,
        0.661618, -2.07873e-14, 1.99471e-14, -0.749841,
        0.999993, -7.98776e-18, -1.52736e-16, -0.00386056,
        0.00876346, -6.10087e-18, -8.96123e-18, -0.999962,
        0.999998, 4.38318e-16, 1.13588e-15, -0.00174402,
        0.000182465, -1.23625e-17, 5.74813e-18, 1,
        1.14556e-05, -2.42153e-17, -8.56192e-18, 1,
        1, -7.38038e-18, 1.93524e-16, 1.77348e-05,
        1.67806e-06, 4.59552e-18, 7.16171e-18, -1,
        0.00421768, -1.0137e-16, -7.9283e-18, -0.999991,
        1, 3.35442e-18, -7.64536e-18, 5.41125e-06,
        0.00552829, -6.20579e-16, -2.39393e-17, -0.999985,
        0.00420983, -1.05895e-17, 2.7528e-18, 0.999991,
        0.707109, -1.59891e-18, -1.2597e-18, 0.707105,
        0.999999, 5.66713e-18, 7.26321e-16, -0.0017136,
        5.03453e-05, -8.81161e-17, -4.79395e-16, 1,
        1, 8.04976e-18, 3.95295e-16, -6.29651e-05,
        0.707146, 2.84033e-16, 3.46871e-16, 0.707068,
        0.99997, 7.34469e-19, 4.61466e-18, -0.00777084,
        0.999999, 8.10821e-18, -1.98681e-17, -0.00147057,
        0.999994, -4.81879e-17, -1.85495e-17, 0.00348654,
        0.999873, -5.55627e-18, 1.58863e-17, -0.0159116,
        0.999959, 7.50535e-18, 1.38572e-17, 0.00908732,
        0.0199596, 8.86484e-19, 5.84366e-19, -0.999801,
        0.00068776, -2.70495e-16, 6.98122e-17, -1,
        0.0130175, 2.77534e-18, 3.65094e-18, 0.999915,
        0.99999, -8.38837e-18, 3.42068e-15, 0.00447051,
        0.999964, -2.15883e-19, -4.12651e-18, -0.00849401,
        0.999998, -2.32472e-17, 4.83193e-16, 0.00212079,
        0.00565214, 5.6354e-17, -7.87419e-18, 0.999984,
        0.0037818, -1.16969e-17, 8.55886e-18, 0.999993,
        0.00604776, -5.12419e-18, -1.71266e-17, -0.999982,
        0.999993, 8.0638e-18, 9.47506e-17, 0.00384074,
        0.00603628, 8.63719e-19, -9.22167e-18, 0.999982,
        0.00382292, 6.82506e-16, 4.89444e-18, 0.999993,
        0.011562, -2.82798e-17, -1.01998e-17, -0.999933,
        0.000104607, 7.39565e-16, 8.1372e-18, -1,
        0.00253685, -5.3812e-17, -1.55105e-17, -0.999997,
        0.0055946, -7.18794e-18, 2.13949e-18, 0.999984,
        0.0039048, -1.04264e-18, 2.98891e-17, -0.999992,
        0.999976, -9.1147e-17, -4.08298e-17, -0.00699168,
        0.00712763, 3.02453e-17, -7.92461e-18, 0.999975,
        0.000626785, 1.41097e-16, 8.51902e-17, -1,
        0.999999, -1.69546e-17, 1.30535e-16, 0.00143954,
        8.64176e-05, -5.46563e-17, -5.36905e-17, -1,
        0.999999, -4.87473e-17, 5.4118e-17, -0.00136986,
        0.707107, 6.29442e-18, -2.28868e-18, 0.707107,
        0.720282, 8.95554e-19, -4.4542e-18, 0.693681,
        0.999998, 7.9831e-18, 3.40137e-16, -0.00192829,
        1, -8.16869e-18, 1.30099e-16, -2.69831e-06,
        0.00477849, -1.7565e-17, -1.2237e-17, -0.999989,
        1, -7.11025e-18, 5.06678e-18, 9.9196e-05,
        0.999949, -8.16538e-18, 4.86964e-17, 0.0100977,
        0.0166897, -1.5545e-18, 8.47923e-18, -0.999861,
        0.0144919, 5.46172e-19, 6.24685e-18, 0.999895,
        1, 5.64325e-17, 5.76522e-17, 0.000512752,
        0.99999, -7.78308e-18, 8.62296e-17, 0.00436859,
        0.707107, 1.60341e-18, 6.57765e-18, 0.707107,
        0.0044663, -5.13799e-17, -1.55579e-16, 0.99999,
        0.0134165, 1.96287e-17, 7.00218e-18, -0.99991,
        0.0021455, 1.50824e-18, -1.68948e-19, 0.999998,
        0.013199, -1.51624e-16, 1.04872e-17, -0.999913,
        0.999973, 9.47191e-18, 1.73425e-17, -0.00732344,
        0.00012749, -4.71037e-18, 8.45715e-18, 1,
        0.707101, -2.3277e-16, 2.20783e-16, -0.707113,
        1, -2.74139e-17, -2.83848e-18, -5.75052e-05,
        0.0051025, -7.33794e-16, 2.90975e-17, -0.999987,
        1, 4.1665e-18, -4.2397e-18, 0.000508038,
        0.999993, 2.1217e-17, 6.15417e-18, -0.00384329,
        1, 7.55884e-18, 2.81186e-17, 0.000977552,
        0.999988, 4.53236e-19, 4.95788e-18, -0.00480025,
        7.91015e-05, 9.91511e-17, 4.42976e-17, 1,
        1, 7.75961e-18, 4.4936e-17, -2.06236e-06,
        0.99998, 8.46453e-18, 2.02173e-18, 0.00637263,
        3.90839e-05, 6.57744e-17, -8.54186e-18, 1,
        1.22966e-05, 1.51212e-16, 8.13277e-18, -1,
        0.00425048, 1.18506e-17, -8.25732e-18, 0.999991,
        0.00583342, -5.16944e-16, 1.4314e-17, -0.999983,
        0.999986, -3.81797e-17, -4.80607e-16, -0.00524882,
        0.000397923, -7.84879e-17, -9.83079e-19, -1,
        0.999996, 3.79005e-17, -1.07698e-16, -0.00278346,
        0.00621585, -1.32508e-18, 8.51491e-18, -0.999981,
        0.999968, -3.87707e-18, 2.22802e-17, -0.00800779,
        0.999988, -4.22456e-19, 1.07982e-18, 0.00498064,
        0.00021947, 3.67425e-17, -2.68681e-17, 1,
        0.707107, -1.38157e-19, -3.8481e-18, -0.707107};//order in w x y z


    double quatMyTotal[] =
    {0.00575856, 0, 0, 0.999983,
        0.0315244, -0, -0, -0.999503,
        0.0103796, -0, -0, -0.999946,
        0.00481234, 0, 0, 0.999988,
        0.999997, 0, 0, 0.00249369,
        0.00788199, -0, -0, -0.999969,
        0.999959, 0, 0, 0.0090045,
        0.0136401, 0, 0, 0.999907,
        0.711822, 0, 0, 0.70236,
        0.999954, 0, 0, -0.00963132,
        0.00787261, -0, -0, -0.999969,
        0.999739, 0, 0, 0.0228339,
        0.999993, 0, 0, 0.0036784,
        0.0167537, -0, -0, -0.99986,
        0.00427742, -0, -0, -0.999991,
        0.999999, 0, 0, 0.00114431,
        0.99999, 0, 0, 0.0044741,
        0.0329528, -0, -0, -0.999457,
        0.999978, 0, 0, 0.00663131,
        0.0291565, -0, -0, -0.999575,
        0.00548129, 0, 0, 0.999985,
        0.707762, 0, 0, 0.706451,
        0.00103612, -0, -0, -0.999999,
        0.0125799, 0, 0, 0.999921,
        0.999997, 0, 0, 0.00260848,
        0.00396178, 0, 0, 0.999992,
        0.00682992, -0, -0, -0.999977,
        0.707117, 0, 0, 0.707096,
        0.708087, 0, 0, 0.706125,
        0.0274573, -0, -0, -0.999623,
        0.999924, 0, 0, 0.0123306,
        0.00523215, 0, 0, 0.999986,
        1, 0, 0, -0.000242357,
        1, 0, 0, 0.000418808,
        0.999606, 0, 0, 0.0280544,
        0.999825, 0, 0, -0.018711,
        0.999463, 0, 0, 0.0327683,
        0.029941, -1.26124e-10, -1.39689e-08, -0.999552,
        0.00372584, -0, -0, -0.999993,
        0.0162683, 0, 0, 0.999868,
        0.00509136, -0, -0, -0.999987,
        0.999916, 0, 0, -0.0129317,
        0.999996, 0, 0, 0.00275142,
        0.000784675, -0, -0, -1,
        0.000488302, -0, -0, -1,
        0.0163202, -0, -0, -0.999867,
        0.999895, 0, 0, 0.0144916,
        0.00471797, 0, 0, 0.999989,
        0.999999, 1.17086e-09, 4.62198e-13, -0.00106364,
        0.0283975, -0, -0, -0.999597,
        0.999999, 0, 0, -0.00156575,
        0.0175737, 0, 0, 0.999846,
        0.00480072, 0, 0, 0.999988,
        0.00921558, -1.83909e-11, -8.6863e-09, -0.999958,
        0.999988, 0, 0, -0.0048215,
        8.16516e-05, 0, 0, 1,
        0.00681514, 0, 0, 0.999977,
        0.999951, 0, 0, 0.00989564,
        0.0156118, -0, -0, -0.999878,
        0.999999, 0, 0, 0.00143479,
        0.707152, 0, 0, 0.707062,
        0.731688, 0, 0, 0.681639,
        0.999999, 0, 0, 0.00106878,
        0.999981, 0, 0, 0.00613471,
        0.0194014, -0, -0, -0.999812,
        0.999958, 0, 0, 0.00913589,
        0.707727, 0, 0, 0.706486,
        0.0255068, -0, -0, -0.999675,
        0.0111217, 0, 0, 0.999938,
        0.999974, 0, 0, 0.00714555,
        0.999976, 0, 0, -0.00694848,
        0.707256, 0, 0, 0.706958,
        0.00330161, -0, -0, -0.999995,
        0.00328831, -0, -0, -0.999995,
        0.00178329, 0, 0, 0.999998,
        0.00433579, 0, 0, 0.999991,
        0.999936, 0, 0, 0.0113533,
        0.014404, 0, 0, 0.999896,
        0.706284, 0, 0, -0.707928,
        0.999855, 0, 0, 0.0170299,
        0.00586975, -0, -0, -0.999983,
        0.999995, 0, 0, -0.00320138,
        0.999757, 0, 0, 0.0220657,
        0.999999, 0, 0, -0.00116253,
        0.707855, 0, 0, 0.706358,
        0.0114428, -0, -0, -0.999935,
        0.706521, 0, 0, 0.707692,
        0.998071, 0, 0, 0.0620808,
        0.00680008, -0, -0, -0.999977,
        0.000216649, -0, -0, -1,
        0.0150056, -0, -0, -0.999887,
        0.0152077, 0, 0, 0.999884,
        0.707711, 0, 0, 0.706502,
        0.00208911, -0, -0, -0.999998,
        0.999967, 0, 0, -0.00810553,
        0.0229764, -0, -0, -0.999736,
        0.999982, 0, 0, -0.00592842,
        0.999841, 0, 0, 0.0178135,
        0.0016269, 0, 0, 0.999999,
        0.706649, 0, 0, -0.707564};//same order
    
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
