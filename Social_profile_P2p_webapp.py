# -*- coding: utf-8 -*-
"""
Created on Sun May 21 09:25:24 2023
"""
import pickle
import streamlit as st
import pandas as pd
import numpy as np

def Application():
    st.title('Social Profile LoanStatus Prediction App by Classification💵')

    model1 = pickle.load(open('model_ensemble-1.pkl', 'rb'))
    model_EMI = pickle.load(open('model_lr_EMI.pkl', 'rb'))
    model_PROI = pickle.load(open('model_lr_PROI.pkl', 'rb'))
    model_ELA = pickle.load(open('model_lr_ELA.pkl', 'rb'))

    def main():
        with st.form('prediction_form'):
            st.header("Predict Loan status from the following features:")
            st.header("![Alt Text](https://media1.tenor.com/images/18d1ad50a584f635b228241edd5f0ba0/tenor.gif?itemid=27008790)")

            BorrowerAPR = st.number_input('BorrowerAPR:') # 1, # 1
            BorrowerRate = st.number_input('BorrowerRate:') # 2
            LenderYield = st.number_input('LenderYield:') # 3
            LoanCurrentDaysDelinquent = st.number_input('LoanCurrentDaysDelinquent:') # 4
            LoanFirstDefaultedCycleNumber = st.number_input('LoanFirstDefaultedCycleNumber:') # 5
            LP_CustomerPayments = st.number_input('LP_CustomerPayments:') # 6, # 4
            LP_CustomerPrincipalPayments = st.number_input('LP_CustomerPrincipalPayments:') # 7, # 5
            LP_GrossPrincipalLoss = st.number_input('LP_GrossPrincipalLoss:') # 8
            LP_NetPrincipalLoss = st.number_input('LP_NetPrincipalLoss:') # 9
            LP_NonPrincipalRecoverypayments = st.number_input('LP_NonPrincipalRecoverypayments:') # 10
            LoanOriginalAmount = st.number_input('LoanOriginalAmount:') # 2
            MonthlyLoanPayment = st.number_input('MonthlyLoanPayment:') # 3
            # LP_CustomerPayments = st.number_input('LP_CustomerPayments:')
            # LP_CustomerPrincipalPayments = st.number_input('LP_CustomerPrincipalPayments:')
            LP_InterestandFees = st.number_input('LP_InterestandFees:') # 6
            LP_ServiceFees = st.number_input('LP_ServiceFees:') # 7
            LoanTenure = st.number_input('LoanTenure:') # 8
            InterestAmount = st.number_input('InterestAmount:') # 9
            # CreditScoreRangeUpper = st.number_input('CreditScoreRangeUpper:')
            TotalAmount = st.number_input('TotalAmount:') # 10
            # AvailableBankcardCredit = st.number_input('AvailableBankcardCredit:')
            # ROI = st.number_input('ROI:')
            # CreditGrade = st.number_input('CreditGrade:')
            # CreditGrade_description = st.number_input('CreditGrade_description:')

            button1 = st.form_submit_button('Predict')

        if button1:
            data1 = np.array([BorrowerAPR, BorrowerRate, LenderYield, LoanCurrentDaysDelinquent, LoanFirstDefaultedCycleNumber,
                             LP_CustomerPayments, LP_CustomerPrincipalPayments, LP_GrossPrincipalLoss, LP_NetPrincipalLoss,
                             LP_NonPrincipalRecoverypayments]).reshape(1, -1)

            prediction1 = model1.predict(data1)
            if prediction1 == [0]:
                st.write('Defaulted')
            else:
                st.write('Not Defaulted')

            st.write(f"The predicted LoanStatus is: {prediction1}")
            
            data2 = np.array([BorrowerRate, LoanOriginalAmount, MonthlyLoanPayment, LP_CustomerPayments,
                    LP_CustomerPrincipalPayments, LP_InterestandFees, LP_ServiceFees, LoanTenure, InterestAmount,
                    TotalAmount]).reshape(1, -1)

            predictionEMI = model_EMI.predict(data2)
            st.write(f"EMI: {predictionEMI}")

            data3= np.array([BorrowerRate, LoanOriginalAmount, MonthlyLoanPayment, LP_CustomerPayments,
                    LP_CustomerPrincipalPayments, LP_InterestandFees, LP_ServiceFees, LoanTenure, InterestAmount,
                    TotalAmount]).reshape(1, -1)
                
            predictionPROI = model_PROI.predict(data3)
            st.write(f"PROI: {predictionPROI}")

            data4 = np.array([BorrowerRate, LoanOriginalAmount, MonthlyLoanPayment, LP_CustomerPayments,
                              LP_CustomerPrincipalPayments, LP_InterestandFees, LP_ServiceFees, LoanTenure,
                              InterestAmount,
                              TotalAmount]).reshape(1, -1)

            predictionELA = model_ELA.predict(data4)
            st.write(f"ELA: {predictionELA}")

    main()

if __name__ == '__main__':
    Application()
