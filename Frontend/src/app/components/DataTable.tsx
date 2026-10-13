import { ReactNode } from "react";

interface Column {
  header: string;
  accessor: string;
  cell?: (value: any, row: any) => ReactNode;
}

interface DataTableProps {
  columns: Column[];
  data: any[];
  className?: string;
}

export function DataTable({ columns, data, className = "" }: DataTableProps) {
  return (
    <div className={`overflow-x-auto ${className}`}>
      <table className="w-full">
        <thead>
          <tr className="border-b border-slate-800/50">
            {columns.map((column) => (
              <th key={column.accessor} className="text-left px-4 py-3 text-sm font-semibold text-slate-300">
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, rowIndex) => (
            <tr key={rowIndex} className="border-b border-slate-800/30 hover:bg-slate-800/20 transition-colors">
              {columns.map((column) => (
                <td key={column.accessor} className="px-4 py-3 text-sm text-slate-400">
                  {column.cell ? column.cell(row[column.accessor], row) : row[column.accessor]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
