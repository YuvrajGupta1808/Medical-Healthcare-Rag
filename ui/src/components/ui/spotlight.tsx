import React from "react";
import { cn } from "@/lib/utils";

export const Spotlight = ({
  className,
  fill = "white",
}: {
  className?: string;
  fill?: string;
}) => {
  return (
    <svg
      className={cn(
        "animate-spotlight pointer-events-none absolute z-[1]  h-[169%] w-[138%] lg:w-[84%] opacity-0",
        className
      )}
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 3787 2842"
      fill="none"
    >
      <g filter="url(#filter-spotlight)">
        <ellipse
          cx="1924.5"
          cy="273.5"
          rx="1924.5"
          ry="273.5"
          transform="matrix(-0.822377 -0.568943 0.568943 -0.822377 1551.54 2277.61)"
          fill={fill}
          fillOpacity="0.21"
        ></ellipse>
      </g>
      <defs>
        <filter
          id="filter-spotlight"
          x="0.860352"
          y="0.838989"
          width="3785.16"
          height="2840.23"
          filterUnits="userSpaceOnUse"
          colorInterpolationFilters="sRGB"
        >
          <feFlood floodOpacity="0" result="BackgroundImageFix"></feFlood>
          <feBlend
            mode="normal"
            in="SourceGraphic"
            in2="BackgroundImageFix"
            result="shape"
          ></feBlend>
          <feGaussianBlur
            stdDeviation="151"
            result="effect1_foregroundBlur_1065_8"
          ></feGaussianBlur>
        </filter>
      </defs>
    </svg>
  );
};
