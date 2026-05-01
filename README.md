# ML Pose Estimation Pipeline for Behavioral Tracking
### The goal of this project is to analyze how hunger impacts social behavior by identifying measurable changes in movement patterns during mouse interactions.
___
This project focuses on a behavioral experiment where one mouse is exposed to a pain stimulus and interacts with a bystander mouse. We compare two conditions: fed vs. food-deprived bystanders, to understand how internal state influences behavior.

###% Problem Statement
Traditional analysis of these interactions relies on manual video scoring, which is time-intensive (~2+ hours per video), difficult to scale, and prone to human bias. To address this, we built an automated pipeline using pose estimation data to track movement and extract behavioral metrics.

#### Objectives
The analysis focuses on two primary objectives:
Identify differences in movement and interaction patterns between fed and food-deprived conditions
Evaluate whether automated tracking can replace manual scoring and uncover consistent behavioral patterns
___
### Insights Summary
___
#### Snapshot 
**Hunger disrupts empathetic social patterns**
Food deprived bystanders show reduced proximity and no trajectory alignment.

**Increased self-directed activity under food deprivation**
Food deprived bystanders exhibit increased exploratory behaviors suggesting physiological neeeds ovveride typical social responses
___

#### Proximity Metrics  
**Behavioral Shift: Fed vs. Food-Deprived Bystanders**

Following the introduction of a pain stimulus, both mice show clear changes in movement patterns. The pain mouse becomes largely stationary (corner-huddling), while fed bystanders reduce exploratory behavior and instead track the movement of the pain mouse.  

In contrast, food-deprived bystanders exhibit increased movement and no longer align with the pain mouse’s trajectory, indicating a disruption in typical social tracking behavior.

<p align="center">
  <img src="" width="100%">
</p>

Despite visible coordination in the fed condition, **no significant difference in physical distance** was observed between fed bystanders and pain mice. However, food-deprived bystanders maintain **significantly greater distance** from their injured partner, suggesting reduced social interaction.

<p align="center">
  <img src="" width="100%">
</p>


#### Velocity Metrics  

In the fed condition, both pain and bystander mice show **reduced movement**, with lower total distance traveled and decreased average velocity during interaction.

<p align="center">
  <img src="" width="100%">
</p>

In contrast, food-deprived bystanders travel **significantly farther and at higher average speeds**, indicating a shift toward increased independent movement.

<p align="center">
  <img src="" width="100%">
</p>


#### Social Interaction Patterns  

Food-deprived bystanders exhibit a higher frequency of **exploratory and high-mobility behaviors** (e.g., walking, turning, running), reflecting a shift away from social engagement toward self-directed activity.

<p align="center">
  <img src="" width="100%">
</p>
___
### Recommendations 
**Expand feature set beyond movement metrics**
Investigate other interaction-specific features (e.g., proximity duration, interaction frequency, behavioral clustering outputs) to better capture the drivers of social behavior since proximity alone did not influence our food-deprived bystander.

**Develop predictive models for behavioral state classification**
Use trajectory and movement features (velocity, distance, spatial alignment) to build models that classify fed vs. food-deprived states, enabling automated detection of behavioral shifts.
