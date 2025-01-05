// Make this function accessible to client-side code
function doGet(e) {
  Logger.log('Query parameters:', e.parameter);
  
  if (e.parameter.page === 'redeem') {
    var template = HtmlService.createTemplateFromFile('Redeem');
    template.passUrl = e.parameter.passUrl;
    template.walletUrl = e.parameter.walletUrl?.replace('http://', 'https://'); // Force HTTPS
    
    return template.evaluate()
      .setTitle('Redeem Coupon')
      .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
  }
  
  return HtmlService.createHtmlOutputFromFile('Index')
    .setTitle('Coupon Generator')
    .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
}
// Add this new function for combined authentication and redemption
function redeemCouponWithAuth(passUrl, credentials) {
  try {
    Logger.log('Attempting to authenticate and redeem coupon');
    
    const spreadsheetId = '1Pb0XyrfEwKsdFonezPrpGjWjEive-8g6qvRsQcOMLlU';
    const sheet = SpreadsheetApp.openById(spreadsheetId).getActiveSheet();
    const data = sheet.getDataRange().getValues();
    
    // Get the indices of the auth columns (last two columns)
    const authorizedEmailColIndex = data[0].length - 2;  // Second to last column
    const authorizedPasswordColIndex = data[0].length - 1;  // Last column
    
    // Find row with matching email and password
    const authRow = data.find(row => 
      row[authorizedEmailColIndex]?.toLowerCase() === credentials.email.toLowerCase() &&
      row[authorizedPasswordColIndex] === credentials.password
    );
    
    if (!authRow) {
      return {
        success: false,
        error: 'Invalid email or password'
      };
    }
    
    // Proceed with coupon redemption
    const rowIndex = data.findIndex(row => {
      const storedUrl = String(row[0]).trim();
      const searchUrl = String(passUrl).trim();
      return storedUrl === searchUrl;
    });
    
    if (rowIndex === -1) {
      return { 
        success: false, 
        error: 'Coupon not found in database'
      };
    }

    const row = data[rowIndex];
    const expiryDate = new Date(row[3]);
    const today = new Date();
    
    if (expiryDate < today) {
      return { success: false, error: 'Coupon has expired' };
    }

    if (row[6] === 'Inactive') {
      return { success: false, error: 'Coupon is no longer active' };
    }

    const uses = row[4];
    let remainingUses = row[5];

    if (uses === 'unlimited') {
      return { success: true, message: 'Coupon redeemed successfully' };
    }

    if (remainingUses <= 0) {
      return { success: false, error: 'No remaining uses for this coupon' };
    }

    remainingUses--;
    sheet.getRange(rowIndex + 1, 6).setValue(remainingUses);

    if (remainingUses === 0) {
      sheet.getRange(rowIndex + 1, 7).setValue('Inactive');
    }

    return { 
      success: true, 
      message: 'Coupon redeemed successfully',
      remainingUses: remainingUses
    };

  } catch (error) {
    Logger.log('Error in redemption process: ' + error.toString());
    return { success: false, error: error.toString() };
  }
}
// This function must be global for client access
// function redeemCoupon(passUrl) {
//   try {
//     Logger.log('Attempting to redeem coupon with URL: ' + passUrl);
    
//     const spreadsheetId = '1Pb0XyrfEwKsdFonezPrpGjWjEive-8g6qvRsQcOMLlU';
//     const sheet = SpreadsheetApp.openById(spreadsheetId).getActiveSheet();
    
//     const data = sheet.getDataRange().getValues();
    
//     Logger.log('First few rows of data:');
//     for (let i = 0; i < Math.min(3, data.length); i++) {
//       Logger.log(`Row ${i}: URL = ${data[i][0]}`);
//     }
    
//     const rowIndex = data.findIndex(row => {
//       const storedUrl = String(row[0]).trim();
//       const searchUrl = String(passUrl).trim();
      
//       Logger.log('Comparing:');
//       Logger.log('Stored URL: ' + storedUrl);
//       Logger.log('Search URL: ' + searchUrl);
      
//       return storedUrl === searchUrl;
//     });
    
//     Logger.log('Found row index: ' + rowIndex);
    
//     if (rowIndex === -1) {
//       return { 
//         success: false, 
//         error: 'Coupon not found in database'
//       };
//     }

//     const row = data[rowIndex];
//     const expiryDate = new Date(row[3]);
//     const today = new Date();
    
//     if (expiryDate < today) {
//       return { success: false, error: 'Coupon has expired' };
//     }

//     if (row[6] === 'Inactive') {
//       return { success: false, error: 'Coupon is no longer active' };
//     }

//     const uses = row[4];
//     let remainingUses = row[5];

//     if (uses === 'unlimited') {
//       return { success: true, message: 'Coupon redeemed successfully' };
//     }

//     if (remainingUses <= 0) {
//       return { success: false, error: 'No remaining uses for this coupon' };
//     }

//     remainingUses--;
//     sheet.getRange(rowIndex + 1, 6).setValue(remainingUses);

//     if (remainingUses === 0) {
//       sheet.getRange(rowIndex + 1, 7).setValue('Inactive');
//     }

//     return { 
//       success: true, 
//       message: 'Coupon redeemed successfully',
//       remainingUses: remainingUses
//     };

//   } catch (error) {
//     Logger.log('Error redeeming coupon: ' + error.toString());
//     return { success: false, error: error.toString() };
//   }
// }
function generateCoupon(formData) {
  try {
    // Add debug logging for incoming date
    Logger.log('Incoming expiry date from form:', formData.expiryDate);
    
    const originalDate = new Date(formData.expiryDate);
    Logger.log('Original date object:', originalDate.toISOString());
    
    const adjustedDate = new Date(originalDate);
    adjustedDate.setDate(adjustedDate.getDate() + 1);
    Logger.log('Adjusted date object:', adjustedDate.toISOString());
    
    // Log what we're actually sending to API
    const dateForAPI = adjustedDate.toISOString().split('T')[0];
    Logger.log('Date being sent to API:', dateForAPI);

    const apiPayload = {
      serviceType: formData.serviceType,
      discount: parseInt(formData.discount),
      expiryDate: dateForAPI
    };

    Logger.log('Full API payload:', JSON.stringify(apiPayload));

    const options = {
      'method': 'post',
      'contentType': 'application/json',
      'payload': JSON.stringify(apiPayload),
      'muteHttpExceptions': true
    };

    let response;
    let attempts = 0;
    const maxAttempts = 5;
    const retryDelay = 2000;

    while (attempts < maxAttempts) {
      response = UrlFetchApp.fetch('https://passapi.onrender.com/generate-pass', options);
      const responseCode = response.getResponseCode();
      
      // Log the API response
      Logger.log('API Response Code:', responseCode);
      Logger.log('API Response Body:', response.getContentText());
      
      if (responseCode === 200) {
        break;
      } else if (responseCode === 502 && attempts < maxAttempts - 1) {
        Utilities.sleep(retryDelay);
        attempts++;
        continue;
      } else {
        throw new Error(`API returned status code: ${responseCode}`);
      }
    }

    const responseData = JSON.parse(response.getContentText());
    
    if (responseData.success && responseData.passUrl) {
      const scriptUrl = ScriptApp.getService().getUrl();
      const securePassUrl = responseData.passUrl.replace('http://', 'https://');
      const redeemUrl = `${scriptUrl}?page=redeem&passUrl=${encodeURIComponent(securePassUrl)}&walletUrl=${encodeURIComponent(securePassUrl)}`;
      
      // Use original expiry date for spreadsheet
      formData.expiryDate = originalDate.toISOString().split('T')[0];
      
      updateCouponSheet(securePassUrl, formData);
      
      return {
        success: true,
        passUrl: securePassUrl,
        redeemUrl: redeemUrl
      };
    } else {
      throw new Error('Invalid response format from server');
    }

  } catch (error) {
    Logger.log('Error in generateCoupon: ' + error.toString());
    return {
      success: false,
      error: error.toString()
    };
  }
}
// function generateCoupon(formData) {
//   try {
//     // Add one day to the expiry date
//     const originalDate = new Date(formData.expiryDate);
//     const adjustedDate = new Date(originalDate);
//     adjustedDate.setDate(adjustedDate.getDate() + 2);
    
//     const apiPayload = {
//       serviceType: formData.serviceType,
//       discount: parseInt(formData.discount),
//       expiryDate: adjustedDate.toISOString().split('T')[0] // Format as YYYY-MM-DD
//     };

//     const options = {
//       'method': 'post',
//       'contentType': 'application/json',
//       'payload': JSON.stringify(apiPayload),
//       'muteHttpExceptions': true
//     };

//     let response;
//     let attempts = 0;
//     const maxAttempts = 5;
//     const retryDelay = 2000;

//     while (attempts < maxAttempts) {
//       response = UrlFetchApp.fetch('https://passapi.onrender.com/generate-pass', options);
//       const responseCode = response.getResponseCode();
      
//       if (responseCode === 200) {
//         break;
//       } else if (responseCode === 502 && attempts < maxAttempts - 1) {
//         Utilities.sleep(retryDelay);
//         attempts++;
//         continue;
//       } else {
//         throw new Error(`API returned status code: ${responseCode}`);
//       }
//     }

//     const responseData = JSON.parse(response.getContentText());
    
//     if (responseData.success && responseData.passUrl) {
//       const scriptUrl = ScriptApp.getService().getUrl();
      
//       // Force HTTPS for the passUrl
//       const securePassUrl = responseData.passUrl.replace('http://', 'https://');
      
//       // Create URLs for QR code with HTTPS
//       const redeemUrl = `${scriptUrl}?page=redeem&passUrl=${encodeURIComponent(securePassUrl)}&walletUrl=${encodeURIComponent(securePassUrl)}`;
      
//       Logger.log('Generated redeem URL: ' + redeemUrl);
      
//       // Use original expiry date when updating spreadsheet
//       formData.expiryDate = originalDate.toISOString().split('T')[0];
      
//       // Update spreadsheet with the HTTPS URL
//       updateCouponSheet(securePassUrl, formData);
      
//       return {
//         success: true,
//         passUrl: securePassUrl,
//         redeemUrl: redeemUrl
//       };
//     } else {
//       throw new Error('Invalid response format from server');
//     }

//   } catch (error) {
//     Logger.log('Error in generateCoupon: ' + error.toString());
//     return {
//       success: false,
//       error: error.toString()
//     };
//   }
// }
function updateCouponSheet(couponUrl, formData) {
  try {
    const spreadsheetId = '1Pb0XyrfEwKsdFonezPrpGjWjEive-8g6qvRsQcOMLlU';
    const sheet = SpreadsheetApp.openById(spreadsheetId).getActiveSheet();
    
    const today = new Date();
    const expiryDate = new Date(formData.expiryDate);
    const status = expiryDate < today ? 'Inactive' : 'Active';
    
    // const rowData = [
    //   couponUrl,
    //   formData.serviceType,
    //   formData.discount,
    //   formData.expiryDate,
    //   formData.uses,
    //   formData.uses,
    //   status
    // ];
    const rowData = [
      couponUrl,
      formData.serviceType,
      formData.discount,
      formData.expiryDate,
      formData.uses,
      formData.uses,
      status,
      formData.email,
      formData.phone
    ];
    
    sheet.appendRow(rowData);
    Logger.log('Successfully updated spreadsheet with coupon data');
    
  } catch (error) {
    Logger.log('Error updating spreadsheet: ' + error.toString());
    throw error;
  }
}
// function sendEmailShare(email, walletUrl) {
//   try {
//     const subject = 'Your Digital Coupon';
//     const body = `Here's your digital coupon! Click the link below to add it to your wallet:\n\n${walletUrl}`;
    
//     MailApp.sendEmail(email, subject, body);
    
//     return { success: true };
//   } catch (error) {
//     Logger.log('Error sending email: ' + error.toString());
//     return { success: false, error: error.toString() };
//   }
// }
function sendEmailShare(email, walletUrl) {
  try {
    const subject = 'Your Digital Coupon';
    const body = `Here's your digital coupon! Click the link below to add it to your wallet:\n\n${walletUrl}`;
    
    MailApp.sendEmail({
      to: email,
      subject: subject,
      body: body
    });
    
    return { success: true };
  } catch (error) {
    Logger.log('Error sending email: ' + error.toString());
    return { success: false, error: error.toString() };
  }
}
function sendSMSShare(phone, walletUrl) {
  try {
    const cleanPhone = phone.replace(/\D/g, '');
    
const carriers = {
  'verizon': `${cleanPhone}@vtext.com`,
  'att': `${cleanPhone}@txt.att.net`,
  'tmobile': `${cleanPhone}@tmomail.net`,
  // 'sprint': `${cleanPhone}@messaging.sprintpcs.com`,
  // 'lycamobile': `${cleanPhone}@lycamobile.com`,   
  'boost': `${cleanPhone}@sms.myboostmobile.com`,
  // 'cricket': `${cleanPhone}@sms.cricketwireless.net`,
  'uscellular': `${cleanPhone}@email.uscc.net`,
  // 'metro': `${cleanPhone}@mymetropcs.com`,
  // 'virgin': `${cleanPhone}@vmobl.com`,
  // 'xfinity': `${cleanPhone}@vtext.com`,
  'googlefi': `${cleanPhone}@msg.fi.google.com`,
  // 'republic': `${cleanPhone}@text.republicwireless.com`,
  // 'straighttalk': `${cleanPhone}@vtext.com`,
  // 'consumercellular': `${cleanPhone}@mailmymobile.net`,
  // 'tracfone': `${cleanPhone}@mmst5.tracfone.com`,
  // 'alltell': `${cleanPhone}@message.alltel.com`,
  // 'simple': `${cleanPhone}@smtext.com`,
  // 'rogers': `${cleanPhone}@pcs.rogers.com`,
  // 'bell': `${cleanPhone}@txt.bell.ca`,
  // 'telus': `${cleanPhone}@msg.telus.com`,
  // 'h2o': `${cleanPhone}@txt.att.net`,
  // 'mint': `${cleanPhone}@tmomail.net`,
  // 'net10': `${cleanPhone}@vtext.com`
};

    
    const message = `Here's your digital coupon! Click to add to wallet: ${walletUrl}`;
    
    for (const carrier in carriers) {
      MailApp.sendEmail({
        to: carriers[carrier],
        subject: 'Apple wallet Coupon',
        body: message
      });
    }
    
    return { success: true };
  } catch (error) {
    Logger.log('Error sending SMS: ' + error.toString());
    return { success: false, error: error.toString() };
  }
}
// Function to send reminder and check current status
function sendExpiryReminder() {
  try {
    const spreadsheetId = '1Pb0XyrfEwKsdFonezPrpGjWjEive-8g6qvRsQcOMLlU';
    const sheet = SpreadsheetApp.openById(spreadsheetId).getActiveSheet();
    const data = sheet.getDataRange().getValues();
    
    // Get tomorrow's date at midnight for comparison
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    tomorrow.setHours(0, 0, 0, 0);
    
    // Skip header row if exists
    const startRow = data[0][0] === 'URL' ? 1 : 0;
    
    // Check each coupon
    for (let i = startRow; i < data.length; i++) {
      const row = data[i];
      const [
        couponUrl,
        serviceType,
        discount,
        expiryDateStr,
        uses,
        remainingUses,
        status,
        email,
        phone
      ] = row;
      
      // Convert expiry date string to Date object for comparison
      const expiryDate = new Date(expiryDateStr);
      expiryDate.setHours(0, 0, 0, 0);
      
      // Check if coupon expires tomorrow AND is active AND has remaining uses
      if (expiryDate.getTime() === tomorrow.getTime() &&
          status === 'Active' &&
          (uses === 'unlimited' || parseInt(remainingUses) > 0)) {
        
        Logger.log(`Sending reminder for coupon expiring tomorrow: ${couponUrl}`);
        Logger.log(`Status: ${status}, Uses: ${uses}, Remaining: ${remainingUses}`);
        
        // Send email reminder
        const emailSubject = '⚠️ Your Coupon Expires Tomorrow!';
        const emailBody = `
Important Reminder: Your ${serviceType} coupon for ${discount}% off expires tomorrow at midnight.

Coupon Details:
- Service: ${serviceType}
- Discount: ${discount}%
- Remaining uses: ${uses === 'unlimited' ? 'Unlimited' : remainingUses}
- Expiry: ${expiryDateStr} at midnight

Don't wait! Use your coupon before it expires: ${couponUrl}

Make sure to use it before midnight tomorrow to not miss out on your savings!`;

        try {
          sendEmailShare(email, couponUrl);
          Logger.log(`Email reminder sent to ${email}`);
        } catch (emailError) {
          Logger.log(`Error sending email: ${emailError}`);
        }
        
        // Send SMS reminder if phone number exists
        if (phone) {
          try {
            const smsMessage = `Reminder: Your ${serviceType} coupon for ${discount}% off expires tomorrow at midnight! Use it before it expires: ${couponUrl}`;
            sendSMSShare(phone, couponUrl);
            Logger.log(`SMS reminder sent to ${phone}`);
          } catch (smsError) {
            Logger.log(`Error sending SMS: ${smsError}`);
          }
        }
      }
    }
  } catch (error) {
    Logger.log(`Error in sendExpiryReminder: ${error}`);
  }
}

// Function to create trigger for 7:00 AM EST check
function setupExpiryCheck() {
  // Delete any existing triggers first
  const triggers = ScriptApp.getProjectTriggers();
  triggers.forEach(trigger => {
    if (trigger.getHandlerFunction() === 'sendExpiryReminder') {
      ScriptApp.deleteTrigger(trigger);
    }
  });
  
  // Create new trigger to run at 7:00 AM EST
  ScriptApp.newTrigger('sendExpiryReminder')
    .timeBased()
    .everyDays(1)
    .atHour(7)
    .inTimezone('America/New_York')
    .create();
    
  Logger.log('Daily expiry check trigger created for 7:00 AM EST');
}

