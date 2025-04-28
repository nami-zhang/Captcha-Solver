chrome.action.onClicked.addListener((tab) => {
  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    files: ['content.js']
  });
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'captureScreenshot') {
    chrome.tabs.captureVisibleTab(null, { format: 'png' }, (dataUrl) => {
      sendResponse({ dataUrl: dataUrl });
    });
    return true; // asynchronous response
  }
  
  if (message.type === 'uploadImage') {
    const dataUrl = message.dataUrl;
    const blob = dataURLtoBlob(dataUrl);
    const formData = new FormData();
    formData.append('image', blob, 'captcha.png');
    
    fetch('http://localhost:5000/solve', {
    // replace with server address if remote
    // remember to modify manifest.json as well
    // fetch('http://18.117.7.118:5000/solve', {
      method: 'POST',
      body: formData
    })
      .then(response => {
        if (!response.ok) throw new Error('Upload failed');
        return response.json();
      })
      .then(data => {
        sendResponse({ success: true, data: data });
      })
      .catch(error => {
        sendResponse({ success: false, error: error.message });
      });
    return true;
  }
  
  if (message.type === "openResultPopup") {
    chrome.windows.create({
      url: chrome.runtime.getURL("result.html"),
      type: "popup",
      width: 400,
      height: 600
    });
    sendResponse({ success: true });
    return true;
  }
});

function dataURLtoBlob(dataurl) {
  const arr = dataurl.split(',');
  const mime = arr[0].match(/:(.*?);/)[1];
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while(n--) {
      u8arr[n] = bstr.charCodeAt(n);
  }
  return new Blob([u8arr], { type: mime });
}
