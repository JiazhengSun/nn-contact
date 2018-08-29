/*
 * Copyright (c) 2015-2016, Graphics Lab, Georgia Tech Research Corporation
 * Copyright (c) 2015-2016, Humanoid Lab, Georgia Tech Research Corporation
 * Copyright (c) 2016, Personal Robotics Lab, Carnegie Mellon University
 * All rights reserved.
 *
 * This file is provided under the following "BSD-style" License:
 *   Redistribution and use in source and binary forms, with or
 *   without modification, are permitted provided that the following
 *   conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *   POSSIBILITY OF SUCH DAMAGE.
 */

#include "dart/dart.hpp"
#include "dart/gui/gui.hpp"
#include "dart/collision/bullet/bullet.hpp"
#include "dart/collision/ode/ode.hpp"
#include <dart/utils/utils.hpp>
#include <iostream>
#include <fstream>

#define NUMSAM 500000
#define UNSYMCONE false

//const char *file = "training_r1_unsym_v1.mat";

using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart;
using namespace utils;
using namespace std;

const char* path;
ofstream file;

class MyWindow : public dart::gui::SimWindow
{
public:
    
    MyWindow(const WorldPtr& world)
    {
        setWorld(world);

        bNode = mWorld->getSkeleton("hopper")->getBodyNode(0);
        
        sampleCount = 0;
        ts = 0;
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
        Eigen::Vector6d pos;
        auto Pi = dart::math::constants<double>::pi();

        double u = dart::math::random(-1,1);
        double theta = dart::math::random(0,2*Pi - 0.05);
        double rd = dart::math::random(0.05, Pi);

        auto exp_x = rd * sqrt(1 - pow(u, 2)) * cos(theta);
        auto exp_y = rd * u;
        auto exp_z = rd * sqrt(1 - pow(u, 2)) * sin(theta);

        double h = 1.0;
        pos << exp_x, exp_y, exp_z, 0,h,0;
        mWorld->getSkeleton("hopper")->getJoint(0)->setPositions(pos);
        mWorld->getSkeleton("hopper")->getJoint(0)->setVelocities(Eigen::Vector6d::Zero());
        
        Eigen::MatrixXd vts(3,8);
        vts << 0.1, 0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1,
               0.05, 0.05, 0.05, 0.05, -0.05, -0.05, -0.05, -0.05, 
               0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1;

        double miny = 999;
        //double miny_x, miny_z;
        //int vt_ind = -1;
        for (int i = 0; i < 8; i++) {
            Eigen::Vector3d vt = vts.col(i);
            Eigen::Vector3d offset(0,0,0);
            auto vt_pos = bNode->getWorldTransform()*(vt + offset);
            if (vt_pos[1]<miny) {
                miny = vt_pos[1];
                //miny_x = vt_pos[0];
                //miny_z = vt_pos[2];
                //vt_ind = i;
            }
        }

        pos << exp_x, exp_y, exp_z, 0,h-(1e-7)-miny,0;
        mWorld->getSkeleton("hopper")->getJoint(0)->setPositions(pos);

        //vel[0]->vel[2] angular vel
        //vel[3]->vel[5] linear vel
        Eigen::Vector6d vel = Eigen::Vector6d::Zero();
        vel[0] = dart::math::random(-20,20);
        vel[1] = dart::math::random(-20,20);
        vel[2] = dart::math::random(-20,20);

        vel[3] = dart::math::random(-5,5); //x vel
        vel[4] = dart::math::random(-6,4); //y vel
        vel[5] = dart::math::random(-5,5); //z vel
        
        // Create reference frames for setting the initial velocity
        Eigen::Isometry3d centerTf(Eigen::Isometry3d::Identity());
        centerTf.translation() = bNode->getSkeleton()->getCOM();
        SimpleFrame center(Frame::World(), "center", centerTf);
        Eigen::Vector3d v = Eigen::Vector3d(vel[3], vel[4], vel[5]);
        Eigen::Vector3d w = Eigen::Vector3d(vel[0], vel[1], vel[2]);
        center.setClassicDerivatives(v, w);
        SimpleFrame ref(&center, "root_reference");
        // 
        ref.setRelativeTransform(bNode->getTransform(&center));
        bNode->getSkeleton()->getJoint(0)->setVelocities(ref.getSpatialVelocity());
        
        //            std::cout << "vel_set:" << bNode->getSpatialVelocity(Frame::World(),Frame::World()).transpose() << std::endl;
        
        VelIn = bNode->getSpatialVelocity(Frame::World(),Frame::World());
        PosIn = bNode->getSkeleton()->getPositions();
        auto oldAng = bNode->getAngularMomentum();
        
        // check collision
        auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
        auto collisionGroup = mWorld->getConstraintSolver()->getCollisionGroup();
        dart::collision::CollisionOption option;
        dart::collision::CollisionResult result;
        bool collision = collisionGroup->collide(option, &result);
        
        if (collision && result.getNumContacts() == 1)
        {
            if (UNSYMCONE)
            {
                auto p1 = result.getContact(0).point;
                auto localp1 =  bNode->getWorldTransform().inverse() * p1;
                Eigen::Vector3d Vcp1;
                Vcp1 = bNode->getLinearVelocity(Eigen::Vector3d(localp1[0], localp1[1], localp1[2]),Frame::World());
                if (Vcp1[0] > 0)
                {
                    bNode->setFrictionCoeff(0.75);
                    mWorld->getSkeleton("ground skeleton")->getBodyNode(0)->setFrictionCoeff(0.75);
                }
                else
                {
                    bNode->setFrictionCoeff(1.5);
                    mWorld->getSkeleton("ground skeleton")->getBodyNode(0)->setFrictionCoeff(1.5);
                }
            }
            
            setbuf(stderr, buf);
            mWorld->getConstraintSolver()->solve();
            if (strlen(buf) > 0)
                {std::cerr << "omit!" << std::endl;}
            else
            {

                // Compute velocity changes given constraint impulses
                for (size_t i=0; i < mWorld->getNumSkeletons(); i++)
                {
                    auto skel = mWorld->getSkeleton(i);
                    if (!skel->isMobile())
                        continue;
                    
                    if (skel->isImpulseApplied())
                    {
                        skel->computeImpulseForwardDynamics();
                        skel->setImpulseApplied(false);
                    }
                }
                
                auto force1 = mWorld->getConstraintSolver()->getLastCollisionResult().getContact(0).force;

                if (force1.norm() < 1e-3)
                {
    //                storeOneInFile(2.0);
                    //Detached case
                }
                else // non-trivial contact impulse
                {
                    auto initialPoint1 = result.getContact(0).point;
                    auto localPoint1 =  bNode->getWorldTransform().inverse() * initialPoint1;
                    
                    Eigen::Vector3d Vcp;
                    Vcp = bNode->getLinearVelocity(Eigen::Vector3d(localPoint1[0], localPoint1[1], localPoint1[2]),Frame::World(),Frame::World());
    //                std::cout << Vcp.transpose() << std::endl;
                    
                    if (Vcp.norm() < 5e-4)
                    {
    //                    storeOneInFile(1.0);
                        //static case
                    }
                    else
                    {
    //                    std::cout << bNode->getConstraintImpulse().transpose() << std::endl;
                        Eigen::Vector3d newVel = bNode->getCOMLinearVelocity(Frame::World(),Frame::World());
                        Eigen::Vector3d newAng = bNode->getAngularMomentum();
                        auto px = newVel[0] - VelIn[3]; // linear impulse in one time step
                        auto pz = newVel[2] - VelIn[5];
                        auto ptheta_y = newAng[1] - oldAng[1]; // Impulse = delMomentum. Momentum = I*w, 
                        storeOneForceInFile(px, pz, ptheta_y);
                    }
                    
                }
            }
            
            buf[0] = '\0';
        }
        else
        {
            std::cout << "invalid: continue" << std::endl;
            SimWindow::timeStepping();
        }
        
        ts++;
    }
    
    void storeOneForceInFile(double px, double pz, double ptheta_y)
    {
        if (sampleCount < NUMSAM)
        {
            cout<<sampleCount<<endl;
            Eigen::Vector3d InOut_1 = Eigen::Vector3d::Zero();
            Eigen::Vector6d InOut_2 = Eigen::Vector6d::Zero();
            Eigen::Vector3d Result = Eigen::Vector3d::Zero();
            InOut_1 << PosIn[0], PosIn[1], PosIn[2]; //theta_z
            InOut_2 <<
                    VelIn[0]/20.0, VelIn[1]/20.0,VelIn[2]/20.0, //angular vels
                    VelIn[3], VelIn[4], VelIn[5]; //linear vels
            Result << px, pz, ptheta_y;
            //cout<<"Here is the result after transpose"<<endl;
            //std::cout << InOut_1.transpose() << std::endl;
            //std::cout << InOut_2.transpose() << std::endl;
            //cout<<Result.transpose()<<endl;
            //cout<<"Here is the result before transpose"<<endl;
            //cout<<InOut<<endl;
            for (int i = 0; i < InOut_1.size(); i++) {
                file<< InOut_1[i];
                file<<",";
            }
            for(int i = 0; i < InOut_2.size(); i++) {
                file<< InOut_2[i];
                file<<",";
            }
            for (int i = 0; i < Result.size(); i++) {
                file<<Result[i];
                file<<",";
            }

            file<<"\n";
            sampleCount++;
        }
        if (sampleCount == NUMSAM)
        {
            file.close();
            printf("Done\n");           
            sampleCount++;
        }
    }
    
    BodyNodePtr bNode;
    Marker* m1;
    
    int ts;
    int sampleCount;
    
    Eigen::Vector6d VelIn;
    Eigen::Vector6d PosIn;
    
    char buf[BUFSIZ];
    
    
protected:
};



int main(int argc, char* argv[])
{
    if (UNSYMCONE) {
        path = "/Users/jiazhengsun/Desktop/nn-contact/data/NN-contact-force/train_data/3D/rect_r1_3D_unsym.csv";   
    } else {
        path = "/Users/jiazhengsun/Desktop/nn-contact/data/NN-contact-force/train_data/3D/rect_r1_3D_sym.csv";
    }
    file.open(path);

    WorldPtr world = SkelParser::readWorld(DART_DATA_PATH"/NN-contact-force/skel/singleBody_genData.skel");
    assert(world != nullptr);
    
    world->setGravity(Eigen::Vector3d(0.0, 0.0, 0));
    
    
    MyWindow window(world);
    while(1){
        window.timeStepping();
    }
    
    // std::cout << "space bar: simulation on/off" << std::endl;

    // glutInit(&argc, argv);
    // window.initWindow(640, 480, "Simple Test");
    // glutMainLoop();
}
