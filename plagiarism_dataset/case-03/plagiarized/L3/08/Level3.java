import java.util.Scanner;
public class Level3 {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        double berat;
        double feet;
        double inches;
        double tinggi;
        double bmi;
        System.out.print("Enter weight in pounds: ");
        berat = in.nextDouble();
        System.out.print("Enter feet: ");
        feet = in.nextDouble();
        System.out.print("Enter inches: ");
        inches = in.nextDouble();
        tinggi = feet * 12 + inches;
        bmi = berat * 0.45359237 / ((tinggi * 0.0254) * (tinggi * 0.0254));
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
