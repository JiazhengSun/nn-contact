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

#define NUMSAM 50000

using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart;
using namespace utils;
using namespace std;

const char* path = "/home/jsun303/Desktop/dart-NN-contact-force/data/NN-contact-force/train_data/poly_c2_sym.csv";
ofstream file(path);

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
       Eigen::MatrixXd vts(3,5);
       vts << 0, 0.2, 0.35, 0.1, -0.05,
       0, 0,   0.15, 0.25, 0.1,
       0, 0,   0,    0,    0;
       
       Eigen::Vector6d pos;
       double theta;
       int vt_ind1, vt_ind2;
       double flag = dart::math::random(0.0,1.0);
       auto Pi = dart::math::constants<double>::pi();
       
       if (flag < 0.2)
       {theta = 1.10714872; vt_ind1 = 4; vt_ind2 = 0;}
       else if (flag < 0.4)
       {theta = 0; vt_ind1 = 0; vt_ind2 = 1;}
       else if (flag < 0.6)
       {theta = 2.35619449019; vt_ind1 = 3; vt_ind2 = 4;}
       else if (flag < 0.8)
       {theta = -2.76108627659; vt_ind1 = 2; vt_ind2 = 3;}
       else
       {theta = -Pi/4; vt_ind1 = 1; vt_ind2 = 2;}
       
       // need to calculate h
       double h = 1.0;
       pos << 0, 0, theta, 0, h, 0;
       mWorld->getSkeleton("hopper")->getJoint(0)->setPositions(pos);
       mWorld->getSkeleton("hopper")->getJoint(0)->setVelocities(Eigen::Vector6d::Zero());
       
       Eigen::Vector3d offset(-0.15, -0.14, 0);
       Eigen::Vector3d vt1 = vts.col(vt_ind1);
       auto vt1_pos = bNode->getWorldTransform() * (vt1+offset);
       Eigen::Vector3d vt2 = vts.col(vt_ind2);
       auto vt2_pos = bNode->getWorldTransform() * (vt2+offset);
       assert(std::abs(vt1_pos[1]-vt2_pos[1])<1e-6);

       pos << 0, 0, theta, 0, h-(1e-7)-vt1_pos[1], 0;
       mWorld->getSkeleton("hopper")->getJoint(0)->setPositions(pos);
       
       
       Eigen::Vector6d vel = Eigen::Vector6d::Zero();
       vel[2] = dart::math::random(-20,20);
       vel[3] = dart::math::random(-5,5);
       vel[4] = dart::math::random(-12,8);
       
       
       // Create reference frames for setting the initial velocity
       Eigen::Isometry3d centerTf(Eigen::Isometry3d::Identity());
       centerTf.translation() = mWorld->getSkeleton("hopper")->getCOM();
       SimpleFrame center(Frame::World(), "center", centerTf);
       Eigen::Vector3d v = Eigen::Vector3d(vel[3], vel[4], 0.0);
       Eigen::Vector3d w = vel[2] * Eigen::Vector3d::UnitZ();
       center.setClassicDerivatives(v, w);
       SimpleFrame ref(&center, "root_reference");
       // ?
       ref.setRelativeTransform(bNode->getTransform(&center));
       mWorld->getSkeleton("hopper")->getJoint(0)->setVelocities(ref.getSpatialVelocity());
       
//            std::cout << "vel_target" << vel.transpose() << std::endl;
//            std::cout << "vel_set:" << bNode->getSpatialVelocity(Frame::World(),Frame::World()).transpose() << std::endl;

       VelIn = bNode->getSpatialVelocity(Frame::World(),Frame::World());
       PosIn = mWorld->getSkeleton("hopper")->getPositions();
       
       // check collision
       auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
       auto collisionGroup = mWorld->getConstraintSolver()->getCollisionGroup();
       dart::collision::CollisionOption option;
       dart::collision::CollisionResult result;
       bool collision = collisionGroup->collide(option, &result);
       
       // permute over all collision points, points must be distinct in x direction to be in C2
       // otherwise, invalid continue
       bool c2flag = false;
       for (int j=0; j<result.getNumContacts(); j++)
       {
           if (std::abs((result.getContact(j).point)[0]-(result.getContact(0).point)[0]) > 0.05)
           {
               c2flag = true;
               break;
           }
       }
       
//        Eigen::Vector3d cPoint1(x1, 0.0, 0.0);
//        auto localPoint1 =  bNode->getWorldTransform().inverse() * cPoint1;
//        Eigen::Vector3d cPoint2(x2, 0.0, 0.0);
//        auto localPoint2 =  bNode->getWorldTransform().inverse() * cPoint2;
       
       
       if (collision && c2flag)
       {
           
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
           
           Eigen::Vector3d newVel = bNode->getLinearVelocity(Frame::World(),Frame::World());

           if ((newVel-VelIn.tail<3>()).norm() < 1e-3)
           {
               storeOneInFile(2.0);
           }
           else // non-trivial contact impulse
           {
               
               Eigen::Vector3d Vcp1, Vcp2;
               Vcp1 = bNode->getLinearVelocity(vt1 + offset, Frame::World(),Frame::World());
               Vcp2 = bNode->getLinearVelocity(vt2 + offset, Frame::World(),Frame::World());
//                std::cout << Vcp1.norm() << " " << Vcp2.norm() << std::endl;
               // almost same here
               if (Vcp1.norm() < 1e-3 && Vcp2.norm() < 1e-3)
               {
                   storeOneInFile(1.0);
               }
               else
               {
                   storeOneInFile(0.0);
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
   
    void storeOneInFile(double label)
    {
        if (sampleCount < NUMSAM)
        {
            std::cout << "a" << (int)label << std::endl;
            double* velarray = VelIn.data();
            //scale down theta_dot by 10
//            velarray[2] = velarray[2]/10.0;
//            double* posarray = PosIn.data();
            Eigen::Vector6d InOut = Eigen::Vector6d::Zero();
            InOut << sin(PosIn[2]), cos(PosIn[2]), VelIn[2]/10.0, VelIn[3], VelIn[4], label;
            //cout<<"Here is the result after transpose"<<endl;
            std::cout << InOut.transpose() << std::endl;
            // cout<<"Here is the result before transpose"<<endl;
            // cout<<InOut<<endl;
            for (int i = 0; i < 6; i++) {
                file<< InOut[i];
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
   
   Eigen::Vector3d initialPoint1;
   Eigen::Vector3d initialPoint2;
   
   char buf[BUFSIZ];
   
protected:
};



int main(int argc, char* argv[])
{
    WorldPtr world = SkelParser::readWorld(DART_DATA_PATH"/skel/mytest/poly5.skel");
   assert(world != nullptr);
   
   world->setGravity(Eigen::Vector3d(0.0, -9.8, 0));
   
   
   MyWindow window(world);
   
   std::cout << "space bar: simulation on/off" << std::endl;
   glutInit(&argc, argv);
   window.initWindow(640, 480, "Simple Test");
   glutMainLoop();
}