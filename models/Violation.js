const { DataTypes, Model } = require('sequelize');
const sequelize = require('./index');

class Violation extends Model {}

Violation.init({
  id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
  clientId: { type: DataTypes.STRING, allowNull: true },
  missingItems: { type: DataTypes.TEXT, allowNull: false }, // JSON array string
  detectedAt: { type: DataTypes.DATE, allowNull: false, defaultValue: DataTypes.NOW },
  frameWidth: { type: DataTypes.INTEGER, allowNull: true },
  frameHeight: { type: DataTypes.INTEGER, allowNull: true }
}, {
  sequelize,
  modelName: 'Violation',
  tableName: 'violations',
  timestamps: false
});

module.exports = Violation;
