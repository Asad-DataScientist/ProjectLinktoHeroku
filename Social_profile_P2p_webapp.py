# -*- coding: utf-8 -*-
"""
Created on Sun May 21 09:25:24 2023

@author: ronit
"""
# http://localhost:8501
# cd C:\Users\Asad Ahmed\AppData\Roaming\JetBrains\PyCharmCE2022.3\scratches
# streamlit run scratch.py [ARGUMENTS]
# heroku logs --tail --app social-profile-p2p-webap

import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np


# Shared by ChatGPT
# Get the port number from the environment variable
port = int(os.environ.get('PORT', 5000))
# Shared by ChatGPT
# Set the configuration options for Streamlit
st.set_option('server.port', port)


import pickle
import streamlit as st
import pandas as pd
import numpy as np
def Classification():
    st.title('Social Profile LoanStatus Prediction App by Classification💵')

    model1 = pickle.load(open('model_ensemble.pkl', 'rb'))

    # Rest of the code...

    def main():
        with st.form('prediction_form'):
            st.header("Predict the input for the following features:") 
            st.header("")
            st.header("![Alt Text](https://i.imgur.com/71KxTWy.jpg)")
            
            BorrowerAPR = st.number_input('BorrowerAPR:', min_value=0.16516)
            BorrowerRate = st.number_input('BorrowerRate:', min_value=0.158)
            LenderYield = st.number_input('LenderYield:', min_value=0.138)
            LoanCurrentDaysDelinquent = st.number_input('LoanCurrentDaysDelinquent:', min_value=140)
            LoanFirstDefaultedCycleNumber = st.number_input('LoanFirstDefaultedCycleNumber:', min_value=8)
            LP_CustomerPayments = st.number_input('LP_CustomerPayments:', min_value=11396.14)
            LP_CustomerPrincipalPayments = st.number_input('LP_CustomerPrincipalPayments:', min_value=3001)
            LP_GrossPrincipalLoss = st.number_input('LP_GrossPrincipalLoss:', min_value=3790.25)
            LP_NetPrincipalLoss = st.number_input('LP_NetPrincipalLoss:', min_value=3790.25)
            LP_NonPrincipalRecoverypayments = st.number_input('LP_NonPrincipalRecoverypayments:', min_value=268.96)
            
            button = st.form_submit_button('Predict')
           
        if button:
            data = np.array([BorrowerAPR, BorrowerRate, LenderYield, LoanCurrentDaysDelinquent, LoanFirstDefaultedCycleNumber,
                             LP_CustomerPayments, LP_CustomerPrincipalPayments, LP_GrossPrincipalLoss, LP_NetPrincipalLoss,
                             LP_NonPrincipalRecoverypayments]).reshape(1, -1)
            
            prediction1 = model1.predict(data)
            if prediction1 == [0]:
                st.write('Loan is not approved')
            else:
                st.write('Loan is approved')
                
            st.write(f"The predicted LoanStatus is: {prediction1}")
            
            
            
    if __name__ == '__main__':
        main()
        
def Regression(): 

    activities = ["EMI","PROI","ELA"]	
    choice = st.sidebar.selectbox("Select Activities",activities)

    if choice == 'EMI':
         st.subheader("Prediction of  EMI")
         
         model_EMI = pickle.load(open('model_lr_EMI.pkl', 'rb'))

         # creating option list for dropdown menu

         features = ['BorrowerRate', 'LoanOriginalAmount', 'MonthlyLoanPayment', 'LP_CustomerPayments', 
                     'LP_CustomerPrincipalPayments', 'LP_InterestandFees', 'LP_ServiceFees', 'LoanTenure', 
                     'InterestAmount', 'TotalAmount']

         st.markdown("<h1 style='color:white ;text-align: center;'>Social Profile LoanStatus Prediction App 💸 </h1>", unsafe_allow_html=True)


         def prediction(num):
             if num ==0:
                 return "Low Intensity"
             elif num==1:
                 return "High Intensity"


         def main():
             with st.form('prediction_form'):

                 st.header("Predict the input for following features:") 
                 st.header("Regression")
                 BorrowerRate = st.number_input('BorrowerRate:', min_value=0.158)
                 LoanOriginalAmount = st.number_input('LoanOriginalAmount:', min_value=10000)
                 MonthlyLoanPayment = st.number_input('MonthlyLoanPayment:', min_value=318.93)
                 LP_CustomerPayments = st.number_input('LP_CustomerPayments:', min_value=1011.65)
                 LP_InterestandFees = st.number_input('LP_InterestandFees:', min_value=0)
                 LP_CustomerPrincipalPayments = st.number_input('LP_CustomerPrincipalPayments:', min_value=1256.63)
                 LP_ServiceFees = st.number_input('LP_ServiceFees:', min_value=-214.04)
                 LoanTenure = st.number_input('LoanTenure:', min_value=24)
                 InterestAmount = st.number_input('InterestAmount:', min_value=1489.150)
                 TotalAmount = st.number_input('TotalAmount:',min_value=10914.1500)
                 button = st.form_submit_button('Predict')
                
             if button:

                 data= np.array([BorrowerRate,LoanOriginalAmount,MonthlyLoanPayment,LP_CustomerPayments, 
                             LP_CustomerPrincipalPayments,LP_InterestandFees,LP_ServiceFees,LoanTenure, 
                             InterestAmount,TotalAmount]).reshape(1, -1)
                 
                 predictionEMI = model_EMI.predict(data)
                     
                 st.write(f"The predicted LoanStatus is:  {predictionEMI}")   

         if __name__ == '__main__':
             main()        
         
    elif choice == 'PROI':
        st.subheader("Prediction of PROI")
        
        model_PROI = pickle.load(open('model_lr_PROI.pkl', 'rb'))

        # creating option list for dropdown menu

        features = ['BorrowerAPR', 'BorrowerRate', 'LenderYield', 'CreditScoreRangeLower', 'CreditScoreRangeUpper',
                    'TotalInquiries', 'AvailableBankcardCredit', 'ROI', 
                    'CreditGrade', 'CreditGrade_description']

        st.markdown("<h1 style='color:white ;text-align: center;'>Social Profile LoanStatus Prediction App 💸 </h1>", unsafe_allow_html=True)


        def prediction(num):
            if num ==0:
                return "Low Intensity"
            elif num==1:
                return "High Intensity"


        def main():
            with st.form('prediction_form'):

                st.header("Predict the input for following features:") 
                st.header("Regression")
                BorrowerAPR = st.number_input('BorrowerAPR:', min_value=0)
                BorrowerRate = st.number_input('BorrowerRate:', min_value=0)
                LenderYield = st.number_input('LenderYield:', min_value=0)
                CreditScoreRangeLower = st.number_input('CreditScoreRangeLower:', min_value=0)
                CreditScoreRangeUpper = st.number_input('CreditScoreRangeUpper:', min_value=0)
                TotalInquiries = st.number_input('TotalInquiries:', min_value=0)
                AvailableBankcardCredit = st.number_input('AvailableBankcardCredit:', min_value=0)
                ROI = st.number_input('ROI:', min_value=0)
                CreditGrade = st.number_input('CreditGrade:', min_value=0)
                CreditGrade_description = st.number_input('CreditGrade_description:', min_value=0)
                
                button = st.form_submit_button('Predict')
               
            if button:

                data= np.array([BorrowerAPR,BorrowerRate,LenderYield,CreditScoreRangeLower,CreditScoreRangeUpper,
                    TotalInquiries, AvailableBankcardCredit, ROI, 
                    CreditGrade,CreditGrade_description]).reshape(1, -1)
                
                predictionPROI = model_PROI.predict(data)
                    
                st.write(f"The predicted LoanStatus is:  {predictionPROI}")   
                
        if __name__ == '__main__':
            main()        
    elif choice == 'ELA':
        st.subheader("Prediction of ELA")
        
        model_ELA = pickle.load(open('model_lr_ELA.pkl', 'rb'))

        # creating option list for dropdown menu

        features = ['LoanOriginalAmount', 'MonthlyLoanPayment', 'LP_CustomerPayments', 
                    'LP_CustomerPrincipalPayments', 'LP_InterestandFees', 'LP_ServiceFees', 
                    'LoanTenure', 'InterestAmount', 'TotalAmount', 'ROI']

        st.markdown("<h1 style='color:white ;text-align: center;'>Social Profile LoanStatus Prediction App 💸 </h1>", unsafe_allow_html=True)


        def prediction(num):
            if num ==0:
                return "Low Intensity"
            elif num==1:
                return "High Intensity"


        def main():
            with st.form('prediction_form'):

                st.header("Predict the input for following features:") 
                st.header("Regression")
                LoanOriginalAmount = st.number_input('LoanOriginalAmount:', min_value=0)
                MonthlyLoanPayment = st.number_input('MonthlyLoanPayment:', min_value=0)
                LP_CustomerPayments = st.number_input('LP_CustomerPayments:', min_value=0)
                LP_CustomerPrincipalPayments = st.number_input('LP_CustomerPrincipalPayments:', min_value=0)
                LP_InterestandFees = st.number_input('LP_InterestandFees:', min_value=0)
                LP_ServiceFees = st.number_input('LP_ServiceFees:', min_value=0)
                LoanTenure = st.number_input('LoanTenure:', min_value=0)
                InterestAmount = st.number_input('InterestAmount:', min_value=0)
                TotalAmount = st.number_input('TotalAmount:', min_value=0)
                ROI = st.number_input('ROI:', min_value=0)
                
                button = st.form_submit_button('Predict')
               
            if button:

                data= np.array([LoanOriginalAmount,MonthlyLoanPayment,LP_CustomerPayments, 
                            LP_CustomerPrincipalPayments, LP_InterestandFees, LP_ServiceFees, 
                            LoanTenure, InterestAmount, TotalAmount, ROI]).reshape(1, -1)
                
                predictionELA = model_ELA.predict(data)
                    
                st.write(f"The predicted LoanStatus is:  {predictionELA}")   
                
        if __name__ == '__main__':
            main()        

page_names_to_funcs = {"Page1" : Classification,"Page2" : Regression}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
