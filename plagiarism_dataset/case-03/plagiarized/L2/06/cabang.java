import java.util.Scanner;
public class cabang {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        System.out.print("Enter weight in pounds : ");
        double w = input.nextDouble();
        System.out.print("Enter feet : ");
        double f = input.nextDouble();
        System.out.print("Enter inches : ");
        double i = input.nextDouble();
        double h = f * 12 + i;
        double TotalBmi = w * 0.45359237 / ((h * 0.0254) * (h * 0.0254)); 
        System.out.println("BMI is " + TotalBmi);
        if (TotalBmi < 18.5) {
            System.out.println("Underweight ");
        } else if (TotalBmi < 25) {
            System.out.println("Normal ");
        } else if (TotalBmi < 30) {
            System.out.println("Overweight ");
        } else {
            System.out.println("Obese ");
        }
    }
}
