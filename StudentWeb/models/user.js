const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
  uid: String,
  name: String,
  matric: String,
  phone: String,
  photo: String
});

module.exports = mongoose.model('User', userSchema);
