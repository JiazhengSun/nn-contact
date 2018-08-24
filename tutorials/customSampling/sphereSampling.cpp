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

#define NUMSAM 30000
#define INWIDTH 5
#define OUTWIDTH 1
#define UNSYMCONE false

using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart;
using namespace utils;
using namespace std;

const char* path;
ofstream file;
int label_0;
int label_1;
int label_2;

class MyWindow : public dart::gui::SimWindow
{
public:
    
    MyWindow(const WorldPtr& world)
    {
        setWorld(world);
        
        bNode = mWorld->getSkeleton("hopper")->getBodyNode(0);
        
        sampleCount = 0;
        ts = 0;

        label_0 = 0;
        label_1 = 0;
        label_2 = 0;
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
        double my_flag = dart::math::random(0.0,1.0);
        auto Pi = dart::math::constants<double>::pi();

        double rd, theta, phi;
        double alpha_y, alpha_a; //a is rotational angle of x or z
        alpha_y = dart::math::random(0.0, 2.0*Pi);
        auto exp_y = 

        // double theta_x, theta_y, theta_z;
        // theta_x = 0; theta_y = 0; theta_z = 0;
        
        // theta_y = dart::math::random(0.0, 2.0*Pi);
        // auto sinY = std::sin(theta_y/2);
        // auto cosY = std::cos(theta_y/2);
        // Eigen::Quaterniond q_y;
        // q_y.x() = 0 * sinY;     q_y.y() = 1 * sinY;     q_y.z() = 0 * sinY;     q_y.w() = cosY;  

        // Eigen::Quaterniond q_next;
        // if(my_flag <= 0.5) {
        //     theta_x = dart::math::random(0,2.0*Pi);
        //     auto sinX = std::sin(theta_x/2);
        //     auto cosX = std::cos(theta_x/2); 
        //     q_next.x() = 1 * sinX;     q_next.y() = 0 * sinX;     q_next.z() = 0 * sinX;     q_next.w() = cosX;         
        // } else {
        //     theta_z = dart::math::random(0,2.0*Pi);
        //     auto sinZ = std::sin(theta_z/2);
        //     auto cosZ = std::cos(theta_z/2);
        //     q_next.x() = 0 * sinZ;     q_next.y() = 0 * sinZ;     q_next.z() = 1 * sinZ;     q_next.w() = cosZ;
        // }

        // Eigen::Quaterniond q_result;
        // q_result = q_y * q_next;

        // Eigen::Vector3d exp_map = dart::math::quatToExp(q_result);

        double h = 1.0;
        pos << exp_map[0], exp_map[1], exp_map[2], 0,h,0;
        mWorld->getSkeleton("hopper")->getJoint(0)->setPositions(pos);
        mWorld->getSkeleton("hopper")->getJoint(0)->setVelocities(Eigen::Vector6d::Zero());
        
        Eigen::MatrixXd vts(3,8);
        vts << 0.1, 0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1,
               0.05, 0.05, 0.05, 0.05, -0.05, -0.05, -0.05, -0.05, 
               0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1;

        double miny = 999;
        for (int i = 0; i < 8; i++) {
            Eigen::Vector3d vt = vts.col(i);
            Eigen::Vector3d offset(0,0,0);
            auto vt_pos = bNode->getWorldTransform()*(vt + offset);
            if (vt_pos[1]<miny) {
                miny = vt_pos[1];
            }
        }

        pos << exp_map[0], exp_map[1], exp_map[2], 0,h-(1e-7)-miny,0;
        mWorld->getSkeleton("hopper")->getJoint(0)->setPositions(pos);
        
        
        Eigen::Vector6d vel = Eigen::Vector6d::Zero();
        vel[0] = dart::math::random(-20,20);
        vel[1] = dart::math::random(-20,20);
        vel[2] = dart::math::random(-20,20);

        vel[3] = dart::math::random(-2,2); //x vel
        vel[4] = dart::math::random(-13,13); //y vel
        vel[5] = dart::math::random(-2,2); //z vel
        
        
        // cosYeate reference frames for setting the initial velocity
        Eigen::Isometry3d centerTf(Eigen::Isometry3d::Identity());
        centerTf.translation() = bNode->getSkeleton()->getCOM();
        SimpleFrame center(Frame::World(), "center", centerTf);
        Eigen::Vector3d v = Eigen::Vector3d(vel[3], vel[4], vel[5]);
        Eigen::Vector3d w = Eigen::Vector3d(vel[0], vel[1], vel[2]);
        center.setClassicDerivatives(v, w);
        SimpleFrame ref(&center, "root_reference");
        // ?
        ref.setRelativeTransform(bNode->getTransform(&center));
        bNode->getSkeleton()->getJoint(0)->setVelocities(ref.getSpatialVelocity());
        
        //            std::cout << "vel_set:" << bNode->getSpatialVelocity(Frame::World(),Frame::World()).transpose() << std::endl;
        
        VelIn = bNode->getSpatialVelocity(Frame::World(),Frame::World());
        PosIn = bNode->getSkeleton()->getPositions();
        
        // check collision
        auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
        auto collisionGroup = mWorld->getConstraintSolver()->getCollisionGroup();
        dart::collision::CollisionOption option;
        dart::collision::CollisionResult result;
        bool collision = collisionGroup->collide(option, &result);
        if (collision && result.getNumContacts() == 2)
        {
            if (UNSYMCONE)
            {
                auto p1 = result.getContact(0).point;
                auto localp1 =  bNode->getWorldTransform().inverse() * p1;
                Eigen::Vector3d VcosX1;
                VcosX1 = bNode->getLinearVelocity(Eigen::Vector3d(localp1[0], localp1[1], localp1[2]),Frame::World());
                if (VcosX1[0] > 0)
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
                auto force2 = mWorld->getConstraintSolver()->getLastCollisionResult().getContact(1).force;
                
                if (force1.norm() < 1e-3 && force2.norm() < 1e-3) // Both points almost no contact, detached.
                {
                    if (label_2 < NUMSAM) {
                        storeOneInFile(2.0);
                        label_2 += 1;
                    }
                }
                else // non-trivial contact impulse
                {
                    auto initialPoint1 = result.getContact(0).point;
                    auto localPoint1 =  bNode->getWorldTransform().inverse() * initialPoint1;
                    
                    auto initialPoint2 = result.getContact(1).point;
                    auto localPoint2 =  bNode->getWorldTransform().inverse() * initialPoint2;
                    
                    Eigen::Vector3d VcosX1, VcosX2;
                    
                    VcosX1 = bNode->getLinearVelocity(Eigen::Vector3d(localPoint1[0], localPoint1[1], localPoint1[2]),Frame::World(),Frame::World());
                    VcosX2 = bNode->getLinearVelocity(Eigen::Vector3d(localPoint2[0], localPoint2[1], localPoint2[2]),Frame::World(),Frame::World());
                    
                    if (VcosX1.norm() < 5e-4 && VcosX2.norm() < 5e-4) // both point's velocity all small, static
                    {
                        if (label_1 < NUMSAM) {
                            storeOneInFile(1.0); // linear velocity small->static case
                            label_1 +=1;
                        }
                    }
                    else
                    {
                        if (label_0 < NUMSAM) {
                            storeOneInFile(0.0); // linear velocity big enough -> dynamic case
                            label_0 +=1;
                        }
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

        if (label_0 == NUMSAM && label_1 == NUMSAM && label_2 == NUMSAM)
        {
            file.close();
            printf("Done\n");           
            sampleCount++;
            cout<<"Label 0 number is: "<<label_0<<endl;
            cout<<"Label 1 number is: "<<label_1<<endl;
            cout<<"Label 2 number is: "<<label_2<<endl;
        }
        
        ts++;
    }
    
    void storeOneInFile(double label)
    {
        cout<<sampleCount<<endl;

        Eigen::Vector6d InOut_1 = Eigen::Vector6d::Zero();
        Eigen::Vector6d InOut_2 = Eigen::Vector6d::Zero();
        InOut_1 << sin(PosIn[0]), cos(PosIn[0]), //theta_x
                sin(PosIn[1]), cos(PosIn[1]), //theta_y
                sin(PosIn[2]), cos(PosIn[2]); //theta_z
        InOut_2 <<
                VelIn[0]/10.0, VelIn[1]/10.0,VelIn[2]/10.0, //angular vels
                VelIn[3], VelIn[4], VelIn[5]; //linear vels
        //cout<<"Here is the result after transpose"<<endl;
        //std::cout << InOut_1.transpose() << std::endl;
        //std::cout << InOut_2.transpose() << std::endl;
        //cout<<label<<endl;
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
        file<<label;
        file<<",";
        file<<"\n";
        sampleCount++;
        
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
        path = "/Users/jiazhengsun/Desktop/nn-contact/data/NN-contact-force/train_data/3D/rect_c2_3D_unsym.csv";   
    } else {
        path = "/Users/jiazhengsun/Desktop/nn-contact/data/NN-contact-force/train_data/3D/rect_c2_3D_sym.csv";
    }
    file.open(path);

    WorldPtr world = SkelParser::readWorld(DART_DATA_PATH"/NN-contact-force/skel/singleBody_genData.skel");
    assert(world != nullptr);
    
    world->setGravity(Eigen::Vector3d(0.0, 0.0, 0));
    
    
    MyWindow window(world);
    
    std::cout << "space bar: simulation on/off" << std::endl;
    
    glutInit(&argc, argv);
    window.initWindow(640, 480, "Simple Test");
    glutMainLoop();
}
