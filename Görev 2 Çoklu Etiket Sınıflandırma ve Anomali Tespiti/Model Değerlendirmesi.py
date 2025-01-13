from pyexpat import model
from sklearn.metrics import classification_report, precision_recall_curve

# Modelin tahminleri
y_pred = model.predict(X_test)

# F1-Score ve Precision-Recall hesaplaması
print(classification_report(y_test, y_pred))

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test.ravel(), y_pred.ravel())
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
