data=readtable("pima_data_diabetes.csv");           
% Replacing zeros with NaN
data.Glucose(data.Glucose == 0) = NaN;
data.BloodPressure(data.BloodPressure == 0) = NaN;
data.SkinThickness(data.SkinThickness == 0) = NaN;
data.Insulin(data.Insulin == 0) = NaN;
data.BMI(data.BMI == 0) = NaN;
data = rmmissing(data);   %removing missing values
features=data{:,1:end-1};
normalized_features=zscore(features);
target=data{:,end};
normalized_data=array2table([normalized_features,target],"VariableNames",data.Properties.VariableNames);
data(:,{'BloodPressure','DiabetesPedigreeFunction'})=[];
cv=cvpartition(size(normalized_data,1),'HoldOut',0.15);
train_data=normalized_data(training(cv),:);
test_data=normalized_data(test(cv),:);
x_train=train_data{:,1:end-1};
y_train=train_data{:,end};
x_test=test_data{:,1:end-1};
y_test=test_data{:,end};
save('preprocessed_pima_data.mat','x_train','y_train','x_test','y_test');
%%
%different models
RF_model=fitcensemble(x_train,y_train,"Method","Bag");
tree_model=fitctree(x_train,y_train);
NB_model=fitcnb(x_train,y_train);
multilayer_perceptron=fitcnet(x_train,y_train,"Standardize",true);
%% predictions
RF_pred=predict(RF_model,x_test);
RF_confusionmatrix=confusionchart(y_test,RF_pred);
tree_pred=predict(tree_model,x_test);
tree_confusionmatrix=confusionchart(y_test,tree_pred);
NB_pred=predict(NB_model,x_test);
NB_confusionmatrix=confusionchart(y_test,NB_pred);
multilayer_pred=predict(multilayer_perceptron,x_test);
multilayer_confusionmatrix=confusionchart(y_test,multilayer_pred);
%%
figure;
subplot(2,2,1)
confusionchart(y_test,RF_pred)
xlabel("Predicted class RF model")
subplot(2,2,2)
confusionchart(y_test,NB_pred)
xlabel("Predicted class NB model")
subplot(2,2,3)
confusionchart(y_test,tree_pred)
xlabel("Predicted class tree model")
subplot(2,2,4)
confusionchart(y_test,multilayer_pred)
xlabel("Predicted class multilayer model")
%%
RF_accuracy=(y_test==RF_pred)/length(y_test);
RF_sensitivity=RF_confusionmatrix(2,2)/(RF_confusionmatrix(2,1)+RF_confusionmatrix(2,2));
RF_specificity=RF_confusionmatrix(1,1)/sum(RF_confusionmatrix(1,:));
NB_accuracy=sum(diag(NB_confusionmatrix)/sum(NB_confusionmatrix(:)));
NB_sensitivity=NB_confusionmatrix(2,2)/sum(NB_confusionmatrix(2,:));
NB_specificity=NB_confusionmatrix(1,1)/sum(NB_confusionmatrix(1,:));
tree_accuracy=sum(diag(tree_confusionmatrix)/sum(tree_confusionmatrix(:)));
tree_sensitivity=tree_confusionmatrix(2,2)/sum(tree_confusionmatrix(2,:));
tree_specificity=tree_confusionmatrix(1,1)/sum(tree_confusionmatrix(1,:));
multilayer_accuracy=sum(diag(multilayer_confusionmatrix)/sum(multilayer_confusionmatrix(:)));
multilayer_sensitivity=multilayer_confusionmatrix(2,2)/sum(multilayer_confusionmatrix(2,:));
multilayer_specificity=multilayer_confusionmatrix(1,1)/sum(multilayer_confusionmatrix(1,:));