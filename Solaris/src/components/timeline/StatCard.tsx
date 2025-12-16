import React from "react";

interface StatCardProps {
  label: string;
  value: string | number;
  subtitle?: string;
}

export const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  subtitle,
}) => {
  return (
    <div className="flex justify-between items-center">
      <span className="text-gray-400 text-sm">{label}</span>
      <div className="text-right">
        <span className="text-2xl font-bold text-white block">{value}</span>
        {subtitle && <span className="text-xs text-amber-500">{subtitle}</span>}
      </div>
    </div>
  );
};
