import java.util.Scanner;
public class Main {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        System.out.print("Enter weight in pounds: ");
        double weights = input.nextDouble();
        System.out.print("Enter feet: ");
        double feets = input.nextDouble();
        System.out.print("Enter inches: ");
        double inches = input.nextDouble();
        double heights = feets * 12 + inches;
        double bmi = weights * 0.45359237
        / ((heights * 0.0254) * (heights * 0.0254));
        System.out.println("BMI is " + bmi);
        if (bmi < 18.5) {
            System.out.println("Underweight");
        } else if (bmi < 25) {
            System.out.println("Normal");
        } else if (bmi < 30) {
            System.out.println("Overweight");
        } else {
            System.out.println("Obese");
        }
    }
}
