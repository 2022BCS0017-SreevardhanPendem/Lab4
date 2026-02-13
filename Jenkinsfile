pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "sreevardhanpendem/wine_predict_2022bcs0017_lab6"
        BUILD_TAG = "${env.BUILD_NUMBER}"
    }

    stages {

        // Stage 1: Checkout
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        // Stage 2: Set up Python Virtual Environment
        stage('Setup Python Virtual Environment') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
        }

        // Stage 3: Train Model
        stage('Train Model') {
            steps {
                sh '''
                . venv/bin/activate
                python train.py
                '''
            }
        }

        // Stage 4: Read Accuracy (R2 + MSE)
        stage('Read Metrics') {
            steps {
                script {
                    def metrics = readJSON file: 'outputs/results.json'
                    env.CURRENT_R2 = metrics.r2_score.toString()
                    env.CURRENT_MSE = metrics.mse.toString()

                    echo "Current R2: ${env.CURRENT_R2}"
                    echo "Current MSE: ${env.CURRENT_MSE}"
                }
            }
        }

        // Stage 5: Compare Accuracy
        stage('Compare Metrics') {
            steps {
                script {
                    withCredentials([
                        string(credentialsId: 'BEST_R2', variable: 'BEST_R2'),
                        string(credentialsId: 'BEST_MSE', variable: 'BEST_MSE')
                    ]) {
        
                        float currentR2 = env.CURRENT_R2.toFloat()
                        float currentMSE = env.CURRENT_MSE.toFloat()
                        float bestR2 = BEST_R2.toFloat()
                        float bestMSE = BEST_MSE.toFloat()
        
                        echo "Best R2: ${bestR2}"
                        echo "Best MSE: ${bestMSE}"
        
                        if (currentR2 > bestR2 && currentMSE < bestMSE) {
                            env.DEPLOY = "true"
                            echo "Model improved → Deployment allowed"
                        } else {
                            env.DEPLOY = "false"
                            echo "Model did not improve → Deployment skipped"
                        }
                    }
                }
            }
        }



        // Stage 6: Build Docker Image (Conditional)
        stage('Build Docker Image') {
            when {
                expression { env.DEPLOY == "true" }
            }
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'docker-hub-token',
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    sh '''
                    echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
                    docker build -t $DOCKER_IMAGE:$BUILD_TAG .
                    docker tag $DOCKER_IMAGE:$BUILD_TAG $DOCKER_IMAGE:latest
                    '''
                }
            }
        }

        // Stage 7: Push Docker Image (Conditional)
        stage('Push Docker Image') {
            when {
                expression { env.DEPLOY == "true" }
            }
            steps {
                sh '''
                docker push $DOCKER_IMAGE:$BUILD_TAG
                docker push $DOCKER_IMAGE:latest
                '''
            }
        }
    }

    // Artifact Archiving (Required)
    post {
        always {
            archiveArtifacts artifacts: 'outputs/**', fingerprint: true
        }
    }
}
