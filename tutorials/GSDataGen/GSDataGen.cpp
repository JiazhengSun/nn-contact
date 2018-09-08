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

//#define NUMSAM 500000
#define NUMSAM 100
#define UNSYMCONE false

using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart;
using namespace utils;
using namespace std;

// Ground truth simluation for error calculations!
class MyWindow : public dart::gui::SimWindow
{
public:
    
    MyWindow(const WorldPtr& world)
    {
        setWorld(world);
        bNode = mWorld->getSkeleton("hand skeleton")->getBodyNode(0);
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
    
    // vel[0] = dart::math::random(-20,20);
    // vel[1] = dart::math::random(-20,20);
    // vel[2] = dart::math::random(-20,20);
    // vel[3] = dart::math::random(-3,3);
    // vel[5] = dart::math::random(-3,3);
    // vel[4] = dart::math::random(-6,3); v_y
    
    void timeStepping() override
    {
        
        if (ts % 800 == 0 && sampleCount < NUMSAM)
        {
            if (ts > 0)
            {
                //cout<<"ts is"<<ts<<endl;
                //Get previous state information
                Eigen::Vector6d EndPos = bNode->getSkeleton()->getPositions();
                EndLinearPos.push_back(Eigen::Vector3d(EndPos[3], EndPos[4], EndPos[5]));
                
                Eigen::Vector3d EndOriExp = Eigen::Vector3d(EndPos[0], EndPos[1], EndPos[2]);
                Eigen::Quaterniond EndOriQuat = dart::math::expToQuat(EndOriExp);
                EndAngularPos.push_back(EndOriQuat);
                sampleCount ++;
                firstContact = true;
            }
            //Set positions
            auto Pi = dart::math::constants<double>::pi();
            Eigen::Vector6d pos = Eigen::Vector6d::Zero();
            pos[0] = dart::math::random(-1*Pi, Pi);
            pos[1] = dart::math::random(-1*Pi, Pi);
            pos[2] = dart::math::random(-1*Pi, Pi);
            pos[3] = 0.0;
            pos[5] = 0.0;
            pos[4] = dart::math::random(0.3, 1.8);
            if (sampleCount < NUMSAM){StartPos.push_back(pos);}
            mWorld->getSkeleton("hand skeleton")->getJoint(0)->setPositions(pos);
            
            // Set velocities
            Eigen::Vector6d vel = Eigen::Vector6d::Zero();
            vel[0] = dart::math::random(-20,20);
            vel[1] = dart::math::random(-20,20);
            vel[2] = dart::math::random(-20,20);
            vel[3] = dart::math::random(-3,3); //x vel
            vel[4] = 0.0; //y vel
            vel[5] = dart::math::random(-3,3); //z vel
            if (sampleCount < NUMSAM){StartVel.push_back(vel);}
            
            // Create reference frames for setting the initial velocity
            Eigen::Isometry3d centerTf(Eigen::Isometry3d::Identity());
            centerTf.translation() = bNode->getSkeleton()->getCOM();
            SimpleFrame center(Frame::World(), "center", centerTf);
            Eigen::Vector3d v = Eigen::Vector3d(vel[3], vel[4], vel[5]);
            Eigen::Vector3d w = Eigen::Vector3d(vel[0], vel[1], vel[2]);
            center.setClassicDerivatives(v, w);
            SimpleFrame ref(&center, "root_reference");
            ref.setRelativeTransform(bNode->getTransform(&center));
            bNode->getSkeleton()->getJoint(0)->setVelocities(ref.getSpatialVelocity());
        }
        else if(ts % 800 == 0 && sampleCount == NUMSAM)
        {
            //cout<<"Total generated sample number is: "<<sampleCount<<endl;
            
            cout<<"Starting positions are: "<<endl;
            for(int i=0; i < StartPos.size(); i++)
            {
                cout<<StartPos[i][0]<<", "<<StartPos[i][1]<<", "<<StartPos[i][2]<<", "<<endl;
            }
            cout<<" "<<endl;
            cout<<"Starting velocities are: "<<endl;
            for(int i=0; i < StartVel.size(); i++)
            {
                cout<<StartVel[i][0]<<", "<<StartVel[i][1]<<", "<<StartVel[i][2]<<", "<<endl;
            }
            cout<<" "<<endl;
            cout<<"Final linear positions are: "<<endl;
            for(int i=0; i < EndLinearPos.size(); i++)
            {
                cout<<EndLinearPos[i][0]<<", "<<EndLinearPos[i][1]<<", "<<EndLinearPos[i][2]<<", "<<endl;
            }
            cout<<" "<<endl;
            cout<<"All end orientation in quaternion in w x y z: "<<endl;
            for(int i=0; i < EndAngularPos.size(); i++)
            {
                Eigen::Quaterniond cur = EndAngularPos[i];
                cout<< cur.w()<<", "<<cur.x()<<", "<<cur.y()<<", "<<cur.z()<<", "<<endl;
            }
            cout<<" "<<endl;
            cout<<"All first contact impulses are: "<<endl;
            for(int i=0; i < FirstContactImpulse.size(); i++) {
                cout<<FirstContactImpulse[i][0]<<", "<<FirstContactImpulse[i][1]<<", "<<FirstContactImpulse[i][2]<<", " << FirstContactImpulse[i][3]<<", "<<FirstContactImpulse[i][4]<<", "<<FirstContactImpulse[i][5]<<", "<<endl;
            }

            sampleCount ++;
        }
        
        // check collision and record impulse
        auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
        auto collisionGroup = mWorld->getConstraintSolver()->getCollisionGroup();
        dart::collision::CollisionOption option;
        dart::collision::CollisionResult result;
        bool collision = collisionGroup->collide(option, &result);
        
        //record impulse of first contact
        ts++;
        SimWindow::timeStepping();
        if (collision && firstContact == true)
        {
            firstContact = false;
            Eigen::Vector6d firstImp = bNode->getConstraintImpulse();
            if (sampleCount < NUMSAM){FirstContactImpulse.push_back(firstImp);}
        }
    } //end of timestepping function
    
    BodyNodePtr bNode;
    int ts;
    int sampleCount;
    bool firstContact = true;
    
    vector<Eigen::Vector6d> StartPos;
    vector<Eigen::Vector6d> StartVel;
    vector<Eigen::Vector3d> EndLinearPos;
    vector<Eigen::Quaterniond> EndAngularPos;
    vector<Eigen::Vector6d> FirstContactImpulse;
    
protected:
};

int main(int argc, char* argv[])
{

    WorldPtr world = SkelParser::readWorld(DART_DATA_PATH"/NN-contact-force/skel/singleBody_test.skel");
    assert(world != nullptr);
    world->setGravity(Eigen::Vector3d(0.0, -10.0, 0));
    MyWindow window(world);
    std::cout << "space bar: simulation on/off" << std::endl;
    glutInit(&argc, argv);
    window.initWindow(640, 480, "Simple Test");
    glutMainLoop();
}
