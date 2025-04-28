document.addEventListener('DOMContentLoaded', () => {
    chrome.storage.local.get('resultData', (data) => {
      if (data.resultData) {
        displayResult(data.resultData);
      } else {
        document.getElementById('resultContent').innerText = 'No result data available.';
      }
    });
  
    document.getElementById('resolve').addEventListener('click', () => {
      chrome.storage.local.get('resultData', (data) => {
        if (data.resultData && data.resultData.croppedDataUrl) {
          uploadCroppedImage(data.resultData.croppedDataUrl);
        }
      });
    });
  });
  
  function displayResult(resultData) {
    const container = document.getElementById('resultContent');
    container.innerHTML = '';
  
    // Display the cropped image.
    const img = document.createElement('img');
    img.src = resultData.croppedDataUrl;
    container.appendChild(img);
  
    // Display predictions.
    const title = document.createElement('p');
    title.innerHTML = '<strong>Predictions:</strong>';
    container.appendChild(title);
  
    const ol = document.createElement('ol');
    resultData.predictions.forEach(item => {
      const li = document.createElement('li');
      
      // Create a text span for prediction and confidence.
      const textSpan = document.createElement('span');
      textSpan.textContent = `${item.prediction} (Confidence: ${(item.confidence * 100).toFixed(2)}%) `;
      li.appendChild(textSpan);
      
      // Create copy button.
      const copyButton = document.createElement('button');
      copyButton.textContent = "Copy";
      copyButton.style.marginLeft = "10px";
      copyButton.addEventListener('click', () => {
        navigator.clipboard.writeText(item.prediction)
          .then(() => {
            copyButton.textContent = "Copied!";
            setTimeout(() => copyButton.textContent = "Copy", 2000);
          })
          .catch(err => {
            console.error("Failed to copy text: ", err);
          });
      });
      
      li.appendChild(copyButton);
      ol.appendChild(li);
    });
    container.appendChild(ol);
  }  
  
  function uploadCroppedImage(dataUrl) {
    const blob = dataURLtoBlob(dataUrl);
    const formData = new FormData();
    formData.append('image', blob, 'captcha.png');
  
    document.getElementById('resultContent').innerText = 'Uploading...';
  
    fetch('http://localhost:5000/solve', {
    //fetch('http://18.117.7.118:5000/solve', {
      method: 'POST',
      body: formData
    })
    .then(response => {
      if (!response.ok) throw new Error('Upload failed');
      return response.json();
    })
    .then(data => {
      const newResult = {
        croppedDataUrl: dataUrl,
        predictions: data.predictions,
        captcha_image: data.captcha_image
      };
      chrome.storage.local.set({ resultData: newResult }, () => {
        displayResult(newResult);
      });
    })
    .catch(error => {
      document.getElementById('resultContent').innerText = 'Error: ' + error.message;
    });
  }
  
  function dataURLtoBlob(dataUrl) {
    const arr = dataUrl.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while(n--) {
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new Blob([u8arr], { type: mime });
  }
  