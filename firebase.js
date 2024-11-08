const admin = require('firebase-admin');

const serviceAccount = require('submissionmlgc-donifitriansyah-c1bba09d520b.json'); // Use the path to your Firebase service account key file

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});

const db = admin.firestore();
module.exports = db;
